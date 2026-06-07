import ray
import wandb
from copy import deepcopy

from agent.workers.DreamerWorker import DreamerWorker, DreamerWorkerBase
# import ipdb

import numpy as np
import pickle
from pathlib import Path
from environments import Env

class DreamerServer:
    def __init__(self, n_workers, env_config, controller_config, model):
        self.env_type = controller_config.ENV_TYPE
        self.use_ray = False
        self.local_worker = None
        self.local_model = model
        
        eval_controller_config = deepcopy(controller_config)
        eval_controller_config.temperature = 1.0  # 1.0
        if hasattr(eval_controller_config, 'determinisitc'):
            eval_controller_config.determinisitc = True

        self.eval_episodes_num = 10
        self.eval_tasks = []
        self.eval_workers = []

        try:
            ray.init()
            self.use_ray = True
        except OSError as exc:
            if n_workers != 1:
                raise RuntimeError(
                    "Ray failed to initialize and local fallback only supports n_workers=1."
                ) from exc

            local_worker_cls = DreamerWorker.__ray_metadata__.modified_class
            self.local_worker = local_worker_cls(0, env_config, controller_config)
            self.eval_workers = [
                local_worker_cls(i, env_config, eval_controller_config)
                for i in range(self.eval_episodes_num)
            ]
            print(f"Ray init failed ({exc}); fallback to local single-process workers.")
            return

        self.workers = [DreamerWorker.remote(i, env_config, controller_config) for i in range(n_workers)]
        self.tasks = [worker.run.remote(model) for worker in self.workers]
        self.eval_workers = [DreamerWorker.remote(i, env_config, eval_controller_config) for i in range(self.eval_episodes_num)]

    def append(self, idx, update):
        if self.use_ray:
            self.tasks.append(self.workers[idx].run.remote(update))
        else:
            self.local_model = update

    def run(self):
        if self.use_ray:
            done_id, tasks = ray.wait(self.tasks)
            self.tasks = tasks
            recvs = ray.get(done_id)[0]
            return recvs

        assert self.local_worker is not None
        return self.local_worker.run(self.local_model)
    
    ## eval
    def eval_append(self, idx, update):
        if self.use_ray:
            self.eval_tasks.append(self.eval_workers[idx].run.remote(update))
        
    def evaluate(self, model_params):
        eval_win_rate = 0.
        eval_returns = 0.
        eval_steps = 0.

        if not self.use_ray:
            for worker in self.eval_workers:
                eval_rollout, eval_info = worker.run(model_params)
                eval_win_rate += eval_info["reward"] if eval_info["reward"] is not None else 0.
                eval_returns += eval_rollout["reward"].sum(0).mean()
                eval_steps += eval_info["steps_done"]

            return (
                eval_win_rate / self.eval_episodes_num,
                eval_returns / self.eval_episodes_num,
                eval_steps / self.eval_episodes_num,
            )
        
        for i in range(self.eval_episodes_num):
            self.eval_append(i, model_params)

        for i in range(self.eval_episodes_num):
            # self.eval_append(i, model_params)
            done_id, eval_tasks = ray.wait(self.eval_tasks)
            
            self.eval_tasks = eval_tasks
            eval_rollout, eval_info = ray.get(done_id)[0]
            
            eval_win_rate += eval_info["reward"] if eval_info["reward"] is not None else 0.
            eval_returns += eval_rollout["reward"].sum(0).mean()
            eval_steps += eval_info["steps_done"]
        
        return eval_win_rate / self.eval_episodes_num, eval_returns / self.eval_episodes_num, eval_steps / self.eval_episodes_num


class LocalDreamerServer:
    def __init__(self, env_config, controller_config, model):
        self.env_type = controller_config.ENV_TYPE
        self.worker = DreamerWorkerBase(0, env_config, controller_config)

        eval_controller_config = deepcopy(controller_config)
        eval_controller_config.temperature = 1.0
        if hasattr(eval_controller_config, "determinisitc"):
            eval_controller_config.determinisitc = True

        self.eval_controller = eval_controller_config.create_controller()
        self.pending_update = model

    def append(self, idx, update):
        del idx
        self.pending_update = update

    def run(self):
        return self.worker.run(self.pending_update)

    def evaluate(self, model_params):
        eval_rollout, eval_info = self.worker.run(model_params, controller=self.eval_controller)
        eval_win_rate = eval_info["reward"] if eval_info["reward"] is not None else 0.0
        eval_returns = eval_rollout["reward"].sum(0).mean()
        eval_steps = eval_info["steps_done"]
        return eval_win_rate, eval_returns, eval_steps


class DreamerRunner:

    def __init__(self, env_config, learner_config, controller_config, n_workers):
        self.n_workers = n_workers
        self.learner = learner_config.create_learner()
        if controller_config.ENV_TYPE == Env.BIDEXHANDS:
            self.server = LocalDreamerServer(env_config, controller_config, self.learner.params())
        else:
            self.server = DreamerServer(n_workers, env_config, controller_config, self.learner.params())

        self.save_path = Path(learner_config.RUN_DIR).parent / f"marie_{learner_config.map_name}_seed{learner_config.seed}.pkl"
        self.env_type = controller_config.ENV_TYPE
        
    def run(self, max_steps=10 ** 10, max_episodes=10 ** 10, save_interval= 10000, save_mode="interval"):
        cur_steps, cur_episode = 0, 0
        last_save_steps = 0
        last_eval_steps = 0
        
        eval_win_rates = []
        eval_ret_list  = []
        steps = []

        wandb.define_metric("steps")
        wandb.define_metric("win", step_metric="steps")
        wandb.define_metric("reward", step_metric="steps")
        wandb.define_metric("rew_per_step", step_metric="steps")
        wandb.define_metric("scores", step_metric="steps")
        wandb.define_metric("returns", step_metric="steps")
        wandb.define_metric("epi_length", step_metric="steps")
        wandb.define_metric("eval_*", step_metric="steps")
        wandb.define_metric("Agent/*", step_metric="steps")
        wandb.define_metric("Model/*", step_metric="steps")
        wandb.define_metric("Value/*", step_metric="steps")
        wandb.define_metric("Policy/*", step_metric="steps")
        wandb.define_metric("world_model/*", step_metric="steps")
        wandb.define_metric("vq/*", step_metric="steps")
        wandb.define_metric("fsq/*", step_metric="steps")

        while True:
            # NOTE: array manager backend... mp
            rollout, info = self.server.run()
            ent = rollout['entropy'].mean(0)
            ent_str = f""
            for e in ent.tolist():
                ent_str += f"{e:.4f} "

            cur_steps += info["steps_done"]
            cur_episode += 1
            epi_length = info["steps_done"]
            returns = rollout["reward"].sum(0).mean()

            if self.env_type == Env.STARCRAFT:
                wandb.log({'win': info["reward"], 'steps': cur_steps})
                print("Epi: %4s" % cur_episode, "steps: %5s" % (cur_steps), f'Win: {info["reward"]}', 'Returns: %.4f' % returns, f"Entropy: {ent_str}", sep=' | ')
            elif self.env_type == Env.SMAX:
                wandb.log({'win': info["reward"], 'steps': cur_steps})
                print("Epi: %4s" % cur_episode, "steps: %5s" % (cur_steps), f'Win: {info["reward"]}', 'Returns: %.4f' % returns, f"Entropy: {ent_str}", sep=' | ')
            elif self.env_type in [Env.MAMUJOCO, Env.PETTINGZOO, Env.BIDEXHANDS]:
                wandb.log({'rew_per_step': info["reward"], 'steps': cur_steps})
                print("Epi: %4s" % cur_episode, "steps: %5s" % (cur_steps), f'Rew per step: {info["reward"]}', 'Returns: %.4f' % returns, f"Average std: {ent_str}", sep=' | ')
            else:
                wandb.log({'scores': info["reward"], 'steps': cur_steps})
                print("Epi: %4s" % cur_episode, "steps: %5s" % (cur_steps), f'Scores: {info["reward"]}', 'Returns: %.4f' % returns, f"Entropy: {ent_str}", sep=' | ')


            wandb.log({'returns': returns, "steps": cur_steps})
            wandb.log({'epi_length': epi_length, "steps": cur_steps})

            self.learner.step(rollout, env_steps=cur_steps)

            ## save model
            if (cur_steps - last_save_steps) >= save_interval and save_mode == "interval":
                self.learner.save(self.learner.config.RUN_DIR + f"/ckpt/model_{cur_steps // 1000}Ksteps.pth")
                last_save_steps = cur_steps // save_interval * save_interval

            ## evaluation
            if (cur_steps - last_eval_steps) >= 1000:
                eval_win_rate, eval_returns, aver_eval_steps = self.server.evaluate(self.learner.params())
                last_eval_steps = cur_steps // 1000 * 1000
                
                wandb.log({'eval_win_rate': eval_win_rate, "steps": cur_steps})
                if self.env_type in [Env.MAMUJOCO, Env.PETTINGZOO, Env.BIDEXHANDS]:
                    wandb.log({'eval_rew_per_step': eval_win_rate, "steps": cur_steps})
                wandb.log({'eval_returns': eval_returns, "steps": cur_steps})
                wandb.log({'eval_avg_epi_len': aver_eval_steps, "steps": cur_steps})

                steps.append(cur_steps)
                eval_win_rates.append(eval_win_rate)
                eval_ret_list.append(eval_returns)

                # Save PKL data immediately after each evaluation
                stored_dict = {
                    'steps': np.array(steps),
                    'eval_win_rates': np.array(eval_win_rates),
                    'eval_returns': np.array(eval_ret_list),
                }
                with open(self.save_path, 'wb') as f:
                    pickle.dump(stored_dict, f)

                if self.env_type == Env.STARCRAFT or self.env_type == Env.SMAX:
                    print(f"Steps: {cur_steps}, Eval_win_rate: {eval_win_rate}, Eval_returns: {eval_returns}, Mean episode length {aver_eval_steps}")

                elif self.env_type in [Env.MAMUJOCO, Env.PETTINGZOO, Env.BIDEXHANDS]:
                    print(f"Steps: {cur_steps}, Eval rew per step: {eval_win_rate}, Eval_returns: {eval_returns}, Mean episode length {aver_eval_steps}")

                else:
                    print(f"Steps: {cur_steps}, Eval average scores: {eval_win_rate}, Eval_returns: {eval_returns}, Mean episode length {aver_eval_steps}")

            if cur_episode >= max_episodes or cur_steps >= max_steps:
                self.learner.save(self.learner.config.RUN_DIR + f"/ckpt/model_final.pth")
                # self.learner.visualize_attention_map(-1, save_mode='final')
                break
            
            self.server.append(info['idx'], self.learner.params())
    
    # only train the actor and critic
    def train_actor(self, world_model_path, max_steps=10 ** 10, max_episodes=10 ** 10):
        ## preload world model
        self.learner.load_pretrained_wm(world_model_path)
        
        cur_steps, cur_episode = 0, 0
        last_save_steps = 0

        wandb.define_metric("steps")
        wandb.define_metric("win rate", step_metric="steps")
        wandb.define_metric("returns", step_metric="steps")
        
        while True:
            rollout, info = self.server.run()
            ent = rollout['entropy'].sum(0) / (rollout['entropy'] > 1e-6).sum(0)
            ent_str = f""
            for e in ent.tolist():
                ent_str += f"{e:.4f} "

            cur_steps += info["steps_done"]
            cur_episode += 1
            returns = rollout["reward"].sum(0).mean()

            wandb.log({'win rate': info["reward"], 'steps': cur_steps})
            wandb.log({'returns': returns, "steps": cur_steps})

            print("%4s" % cur_episode, "%5s" % (cur_steps), info["reward"], 'Returns: %.4f' % returns, f"Entropy: {ent_str}", sep=' | ')

            # train actor only
            self.learner.train_actor_only(rollout)

            if (cur_steps - last_save_steps) > 10000:
                self.learner.save(self.learner.config.RUN_DIR + f"/ckpt/model_{cur_steps // 10000}Ksteps.pth")
                last_save_steps = cur_steps // 10000 * 10000

            if cur_episode >= max_episodes or cur_steps >= max_steps:
                break
            
            self.server.append(info['idx'], self.learner.params())
        
        
