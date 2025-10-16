#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from benchmarl.hydra_config import load_experiment_from_hydra

from benchmarl.experiment.callback import Callback
from tensordict import TensorDict, TensorDictBase
from benchmarl.experiment.logger import JsonWriter
from typing import List
import torch
from scipy.spatial import distance
import os

def get_algo_and_task_name(original_task_name, original_algo_name, group_name=''):
    method_name = ''

    if original_task_name.endswith('_ia'):
        method_name = ' w/ IA'
    elif original_task_name.endswith('_svo'):
        method_name = ' w/ SVO'
    elif original_task_name.endswith('_flsvo'):
        method_name = ' w/ Fair&LocalSVO'
    elif original_task_name.endswith('_flia'):
        method_name = ' w/ Fair&LocalIA'

    algo_name = f'{original_algo_name.upper()}{method_name}'
    task_name = original_task_name.replace('_ia', '').replace('_svo', '').replace('_flia', '').replace('_flsvo', '')

    if 'player_' in group_name: # single player
        player_indx = int(group_name.split('player_')[-1])
        return f'{algo_name} (Agent {player_indx})', task_name
    elif 'players_r' in group_name: # group of players by type
        role_indx = int(group_name.split('players_r')[-1])
        if 'commons_harvest' in original_task_name:
            if role_indx == 0:
                return f'{algo_name} - Standard', task_name
            elif role_indx == 1:
                return f'{algo_name} - High-reward', task_name
            elif role_indx == 2:
                return f'{algo_name} - Wide-zap', task_name
            elif role_indx == 3:
                return f'{algo_name} - Low-reward', task_name
            elif role_indx == 4:
                return f'{algo_name} - Strong (+0.5 Per Zap Hit)', task_name
            elif role_indx == 5:
                return f'{algo_name} - Strong (Spawns Inside)', task_name
            elif role_indx == 6:
                return f'{algo_name} - Variable Speed', task_name
            elif role_indx == 7:
                return f'{algo_name} - High-reward Cooperator', task_name
            elif role_indx == 8:
                return f'{algo_name} - Low-reward Cooperator', task_name
            elif role_indx == 9:
                return f'{algo_name} - Standard Cooperator', task_name
            else:
                return algo_name, task_name
        elif 'clean_up' in original_task_name:
            if role_indx == 0:
                return f'{algo_name} - Standard', task_name
            elif role_indx == 1:
                return f'{algo_name} - High-reward', task_name
            elif role_indx == 2:
                return f'{algo_name} - Wide-zap', task_name
            elif role_indx == 3:
                return f'{algo_name} - Low-reward', task_name
            elif role_indx == 4:
                return f'{algo_name} - Strong (+0.5 Per Zap Hit)', task_name
            elif role_indx == 5:
                return f'{algo_name} - Weak (-0.1 per Clean Hit)', task_name
            elif role_indx == 6:
                return f'{algo_name} - Short-clean', task_name
            elif role_indx == 7:
                return f'{algo_name} - High-reward Cooperator', task_name
            elif role_indx == 8:
                return f'{algo_name} - Low-reward Cooperator', task_name
            elif role_indx == 9:
                return f'{algo_name} - Standard Cooperator', task_name
            elif role_indx == 10:
                return f'{algo_name} - Short-clean Cooperator', task_name
            else:
                return algo_name, task_name
        elif 'coins' in original_task_name:
            if role_indx == 0:
                return f'{algo_name} - Standard Blue', task_name
            elif role_indx == 1:
                return f'{algo_name} - Standard Red', task_name
            elif role_indx == 2:
                return f'{algo_name} - Low-reward Blue', task_name
            elif role_indx == 3:
                return f'{algo_name} - High-reward Red', task_name
            elif role_indx == 4:
                return f'{algo_name} - Slow-move Blue', task_name
            elif role_indx == 5:
                return f'{algo_name} - Spawn-biased Red', task_name
            elif role_indx == 6:
                return f'{algo_name} - Low-reward Blue Cooperator', task_name
            elif role_indx == 7:
                return f'{algo_name} - High-reward Red Cooperator', task_name
            elif role_indx == 8:
                return f'{algo_name} - Standard Blue Cooperator', task_name
            elif role_indx == 9:
                return f'{algo_name} - Spawn-biased Red Cooperator', task_name
            else:
                return algo_name, task_name
        else:
            return algo_name, task_name
    else:
        return algo_name, task_name

class SSDCallback(Callback):
    def on_setup(self):
        self.json_writers = {}
        if self.experiment.config.create_json:
            # a json writer for each group (for each player in melting pot)
            for group in self.experiment.group_map.keys():
                algo_name, task_name = get_algo_and_task_name(self.experiment.task_name, self.experiment.algorithm_name, group)
                self.json_writers[group] = JsonWriter(
                    folder=self.experiment.folder_name,
                    name=self.experiment.name + f"_ssd_{group}.json",
                    algorithm_name=algo_name,
                    task_name=task_name,
                    environment_name=self.experiment.environment_name,
                    seed=self.experiment.seed,
                )

            algo_name, task_name = get_algo_and_task_name(self.experiment.task_name, self.experiment.algorithm_name)
            # a json writer for all the players
            self.json_writers['players'] = JsonWriter(
                folder=self.experiment.folder_name,
                name=self.experiment.name + "_ssd_players.json",
                algorithm_name=algo_name,
                task_name=task_name,
                environment_name=self.experiment.environment_name,
                seed=self.experiment.seed,
            )

    def _get_apples_per_agent(self, td: TensorDictBase):
        return td.get("observation")["PLAYER_ATE_APPLE"].sum(0).mean().item()

    def _get_zap_per_agent(self, td: TensorDictBase):
        zap = td.get("observation")["PLAYER_CALLED_ZAP"]  # Shape: [T, N]
        is_zapped = td.get("observation")["PLAYER_IS_ZAPPED"].squeeze(-1)  # Shape: [T, N]

        T, N = zap.shape[:2]
        zap_frequencies = torch.zeros(N, device=zap.device)

        for i in range(N):
            agent_called_zap = zap[:, i]  # Shape: [N]
            agent_is_zapped = is_zapped[:, i]  # Shape: [N]

            # Filter positions for agents that are not zapped
            alive_called_zap = agent_called_zap[agent_is_zapped == 0]
            zap_frequencies[i] = alive_called_zap.sum() / (alive_called_zap.size(0) + 1)

        return zap_frequencies.mean().item()  # Average across agents

    def _get_peace(self, td: TensorDictBase):
        T = td.get("observation").shape[0] # number of time steps in the rollout
        N = td.get("observation").shape[1] # number of agents
        return N - (td.get("observation")["PLAYER_IS_ZAPPED"].sum(0).sum(0) / T)

    def _get_sustainability(self, td: TensorDictBase):
        T = td.get("observation").shape[0] # number of time steps in the rollout
        N = td.get("observation").shape[1] # number of agents
        sustainability = 0
        for i in range(N):
            ti = torch.where(td.get("observation")["PLAYER_ATE_APPLE"][:,i,:] == 1)[0] # the time steps where agent i collected an apple
            if ti.shape[0] > 0:
            	sustainability += ti.sum() / ti.shape[0]
        return sustainability / N

    def _get_equality(self, td: TensorDictBase):
        rewards = td.get("reward")
        T, N = rewards.shape[:2]  # Number of time steps and agents
        if N == 1:
            return 1
        agent_returns = rewards.sum(0)
        collective_return = agent_returns.sum()

        # Vectorized approach to compute pairwise differences
        abs_differences = torch.abs(agent_returns[:, None] - agent_returns).sum()

        # Compute equality using the optimized sum
        equality = 1 - (abs_differences / (2 * N * (collective_return + 1e-5)))
        return equality.item()

    def _get_equality_ext_return(self, td: TensorDictBase):
        rewards = td.get("observation")["EXT_REWARD"]
        T, N = rewards.shape[:2]  # Number of time steps and agents
        if N == 1:
            return 1
        agent_returns = rewards.sum(0)
        collective_return = agent_returns.sum()

        # Vectorized approach to compute pairwise differences
        abs_differences = torch.abs(agent_returns[:, None] - agent_returns).sum()

        # Compute equality using the optimized sum
        equality = 1 - (abs_differences / (2 * N * (collective_return + 1e-5)))
        return equality.item()

    def _get_ext_return_per_agent(self, td: TensorDictBase):
        return td.get("observation")["EXT_REWARD"].sum(0).mean().item()

    def _get_last_tsr_range_per_agent(self, td: TensorDictBase):
        return td.get("observation")["TSR_RANGE"][-1].mean().item()

    def _get_average_age_per_agent(self, td: TensorDictBase):
        return td.get("observation")["TOTAL_AGE"].mean(0).mean().item()

    def _get_ext_return_coef_var(self, td: TensorDictBase):
        rewards = td.get("observation")["EXT_REWARD"].sum(0)
        if rewards.numel() > 1:
            mean = rewards.mean().item()
            std = rewards.std().item()
            return std / (mean + 1e-8)
        return 0

    def _get_equality_by_apples(self, td: TensorDictBase):
        agent_num_apples = td.get("observation")["PLAYER_ATE_APPLE"]
        T, N = agent_num_apples.shape[:2]  # Number of time steps and agents
        if N == 1:
            return 1
        agent_apples = agent_num_apples.sum(0)
        collective_apples = agent_apples.sum()

        # Vectorized approach to compute pairwise differences
        abs_differences = torch.abs(agent_apples[:, None] - agent_apples).sum()

        # Compute equality using the optimized sum
        equality = 1 - (abs_differences / (2 * N * (collective_apples + 1e-5)))
        return equality.item()

    def _get_average_zapped_others(self, td: TensorDictBase):
        return td.get("observation")["NUM_OTHERS_PLAYER_ZAPPED_THIS_STEP"].sum(0).mean().item()

    def _get_prop_own_coins(self, td: TensorDictBase):
        matched = td.get("observation")["MATCHED_COIN_COLLECTED"].sum(0)  # [N]
        mismatched = td.get("observation")["MISMATCHED_COIN_COLLECTED"].sum(0)  # [N]
        total = matched + mismatched  # [N]
        proportion = torch.where(total > 0, matched / total, torch.full_like(total, 0.5))  # [N]
        return proportion.mean().item()

    def _get_average_collected_coins(self, td: TensorDictBase):
        matched = td.get("observation")["MATCHED_COIN_COLLECTED"].sum(0)  # [N]
        mismatched = td.get("observation")["MISMATCHED_COIN_COLLECTED"].sum(0)  # [N]
        total = matched + mismatched  # [N]
        return total.mean().item()  # scalar: avg coins per player

    def _get_average_mismatch_coins(self, td: TensorDictBase):
        mismatched = td.get("observation")["MISMATCHED_COIN_COLLECTED"].sum(0)  # [N]
        return mismatched.mean().item()

    def _get_intra_group_distance(self, td: TensorDictBase):
        obs = td.get("observation")
        positions = obs["POSITION"].float()  # Shape: [T, N, 2]

        T, N = positions.shape[:2]  # Time steps and number of agents

        is_zapped = (
            obs["PLAYER_IS_ZAPPED"].squeeze(-1)
            if "PLAYER_IS_ZAPPED" in obs
            else torch.zeros((T, N), dtype=torch.bool, device=obs.device)
        )

        total_avg_distance = 0
        valid_timesteps = 0

        if N == 1:
            return 0

        for t in range(T):
            # Get positions and zapped status for all agents at time t
            timestep_positions = positions[t]  # Shape: [N, 2]
            timestep_is_zapped = is_zapped[t]  # Shape: [N]

            # Filter positions for agents that are not zapped
            alive_positions = timestep_positions[timestep_is_zapped == 0]  # Shape: [M, 2], M <= N

            if alive_positions.size(0) > 1:  # At least two agents must be alive
                # Compute pairwise distances
                distances = torch.cdist(alive_positions, alive_positions, p=2)  # Shape: [M, M]

                # Compute the mean of the upper triangle (excluding diagonal)
                avg_distance = distances.sum().item() / (alive_positions.size(0) * (alive_positions.size(0) - 1))

                total_avg_distance += avg_distance
                valid_timesteps += 1

        # Average distance across all valid time steps
        return total_avg_distance / valid_timesteps if valid_timesteps > 0 else 0


    def _get_coverage_per_agent(self, td: TensorDictBase):
        obs = td.get("observation")
        positions = obs["POSITION"]  # Shape: [T, N, 2]

        T, N = positions.shape[:2]  # Time steps and number of agents

        is_zapped = (
            obs["PLAYER_IS_ZAPPED"].squeeze(-1)
            if "PLAYER_IS_ZAPPED" in obs
            else torch.zeros((T, N), dtype=torch.bool, device=obs.device)
        )

        num_unique_positions = 0

        for i in range(N):
            # Get positions and zapped status for agent i
            agent_positions = positions[:, i]  # Shape: [T, 2]
            agent_is_zapped = is_zapped[:, i]  # Shape: [T]

            # Filter out positions where the agent is zapped
            alive_positions = agent_positions[agent_is_zapped == 0]

            # Count unique positions only for alive instances
            num_unique_positions += torch.unique(alive_positions, dim=0).size(0)

        # Average unique positions per agent
        return num_unique_positions / N

    def _write_metrics(self, group, rollouts, writer):
        json_metrics = {}

        if "observation" in rollouts[0].keys(True, True): # test the keys only in the first rollout as the rest will have the same set of keys
            if "PLAYER_IS_ZAPPED" in rollouts[0].get("observation"):
                peaces = [self._get_peace(td) for td in rollouts]
                json_metrics["peace"] = torch.tensor(peaces, device=rollouts[0].device)

            if "PLAYER_CALLED_ZAP" in rollouts[0].get("observation"):
                zap_per_agents = [self._get_zap_per_agent(td) for td in rollouts]
                json_metrics["zap_per_agent"] = torch.tensor(zap_per_agents, device=rollouts[0].device)

                if "NUM_OTHERS_PLAYER_ZAPPED_THIS_STEP" in rollouts[0].get("observation"):
                    average_zapped_others = [self._get_average_zapped_others(td) for td in rollouts]
                    json_metrics["average_zapped_others"] = torch.tensor(average_zapped_others, device=rollouts[0].device)

            if "PLAYER_ATE_APPLE" in rollouts[0].get("observation"):
                sustainabilities = [self._get_sustainability(td) for td in rollouts]
                apples_per_agents = [self._get_apples_per_agent(td) for td in rollouts]

                json_metrics["sustainability"] = torch.tensor(sustainabilities, device=rollouts[0].device)
                json_metrics["apples_per_agent"] = torch.tensor(apples_per_agents, device=rollouts[0].device)

            if "MATCHED_COIN_COLLECTED" in rollouts[0].get("observation") and "MISMATCHED_COIN_COLLECTED" in rollouts[0].get("observation"):
                proportions_own_coins = [self._get_prop_own_coins(td) for td in rollouts]
                average_collected_coins = [self._get_average_collected_coins(td) for td in rollouts]
                average_mismatch_coins = [self._get_average_mismatch_coins(td) for td in rollouts]

                json_metrics["proportion_own_coins"] = torch.tensor(proportions_own_coins, device=rollouts[0].device)
                json_metrics["average_collected_coins"] = torch.tensor(average_collected_coins, device=rollouts[0].device)
                json_metrics["average_mismatch_coins"] = torch.tensor(average_mismatch_coins, device=rollouts[0].device)

            if "POSITION" in rollouts[0].get("observation"):
                intra_group_distances = [self._get_intra_group_distance(td) for td in rollouts]
                coverages = [self._get_coverage_per_agent(td) for td in rollouts]

                json_metrics["intra_group_distance"] = torch.tensor(intra_group_distances, device=rollouts[0].device)
                json_metrics["coverage_per_agent"] = torch.tensor(coverages, device=rollouts[0].device)

            if "EXT_REWARD" in rollouts[0].get("observation"):
                ext_returns = [self._get_ext_return_per_agent(td) for td in rollouts]
                ext_return_coef_vars = [self._get_ext_return_coef_var(td) for td in rollouts]
                ext_return_equalities = [self._get_equality_ext_return(td) for td in rollouts]

                json_metrics["ext_return"] = torch.tensor(ext_returns, device=rollouts[0].device)
                json_metrics["ext_return_coef_var"] = torch.tensor(ext_return_coef_vars, device=rollouts[0].device)
                json_metrics["ext_return_equality"] = torch.tensor(ext_return_equalities, device=rollouts[0].device)

            if "TOTAL_AGE" in rollouts[0].get("observation"):
                average_ages = [self._get_average_age_per_agent(td) for td in rollouts]

                json_metrics["average_age"] = torch.tensor(average_ages, device=rollouts[0].device)

        writer.write(
            metrics=json_metrics,
            total_frames=self.experiment.total_frames,
            evaluation_step=self.experiment.total_frames // self.experiment.config.evaluation_interval,
        )

    def _write_collection_metrics(self, group, rollouts, writer):
        json_metrics = {}

        if "observation" in rollouts[0].keys(True, True): # test the keys only in the first rollout as the rest will have the same set of keys
            if "TSR_RANGE" in rollouts[0].get("observation"):
                last_ranges = [self._get_last_tsr_range_per_agent(td) for td in rollouts]
                json_metrics["last_tsr_range"] = torch.tensor(last_ranges, device=rollouts[0].device)

        writer.write(
            metrics=json_metrics,
            total_frames=self.experiment.total_frames,
            evaluation_step=self.experiment.total_frames // self.experiment.config.evaluation_interval,
        )

    def on_batch_collected(self, batch: TensorDictBase):
        if (self.experiment.config.evaluation and (self.experiment.total_frames % self.experiment.config.evaluation_interval == 0 or self.experiment.n_iters_performed == 0) and (len(self.experiment.config.loggers) or self.experiment.config.create_json)):
            roles = {}

            # Convert batch to list of rollout TensorDicts (one per env)
            rollouts = list(batch.unbind(0))  # list of [n_steps, ...] TensorDicts

            for group in self.experiment.group_map.keys():
                # single player for each group
                rollouts_group = [td.get(("next", group)) for td in rollouts]
                self._write_collection_metrics(group, rollouts_group, self.json_writers[group])

                if ("next", group, "observation", "PLAYER_ROLE_INDEX") in rollouts[0].keys(True, True):
                    player_role = int(rollouts[0].get(("next", group, "observation", "PLAYER_ROLE_INDEX"))[0][0].item())
                    if player_role not in roles.keys():
                        roles[player_role] = []
                    roles[player_role].append(group)

            # collect the commons keys among groups before stacking
            groups = list(self.experiment.group_map.keys())
            shared_keys_observation = set(rollouts[0].get(("next", groups[0], 'observation')).keys()).intersection(*[rollouts[0].get(("next", group, 'observation')).keys() for group in groups[1:]])

            rollouts_all = []
            for td in rollouts:
                groups_dict_list = []
                for group in self.experiment.group_map.keys():
                    group_shared = TensorDict({
                        'observation': td.get(("next", group, 'observation')).select(*shared_keys_observation),
                        'reward': td.get(("next", group, 'reward'))
                    }, batch_size=td.batch_size)
                    groups_dict_list.append(group_shared)
                rollouts_all.append(torch.stack(groups_dict_list, dim=1))
            self._write_collection_metrics('players', rollouts_all, self.json_writers['players'])
            del rollouts_all
            if len(roles.keys()) > 1:
                for r in roles.keys():
                    writer_key = f"players_r{r}"
                    if writer_key not in self.json_writers.keys():
                        algo_name, task_name = get_algo_and_task_name(self.experiment.task_name, self.experiment.algorithm_name, writer_key)
                        self.json_writers[writer_key] = JsonWriter(
                            folder=self.experiment.folder_name,
                            name=f"{self.experiment.name}_ssd_{writer_key}.json",
                            algorithm_name=algo_name,
                            task_name=task_name,
                            environment_name=self.experiment.environment_name,
                            seed=self.experiment.seed,
                        )

                    rollouts_role = []
                    for td in rollouts:
                        groups_dict_list = []
                        for group in roles[r]:
                            groups_dict_list.append(td.get(("next", group)))
                        rollouts_role.append(torch.stack(groups_dict_list, dim=1))

                    self._write_collection_metrics(group, rollouts_role, self.json_writers[writer_key])

    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        if self.experiment.config.create_json:
            roles = {}
            for group in self.experiment.group_map.keys():
                # single player for each group
                rollouts_group = [td.get(("next", group)) for td in rollouts]
                self._write_metrics(group, rollouts_group, self.json_writers[group])

                if ("next", group, "observation", "PLAYER_ROLE_INDEX") in rollouts[0].keys(True, True):
                    player_role = int(rollouts[0].get(("next", group, "observation", "PLAYER_ROLE_INDEX"))[0][0].item())
                    if player_role not in roles.keys():
                        roles[player_role] = []
                    roles[player_role].append(group)

            # collect the commons keys among groups before stacking
            groups = list(self.experiment.group_map.keys())
            shared_keys_observation = set(rollouts[0].get(("next", groups[0], 'observation')).keys()).intersection(*[rollouts[0].get(("next", group, 'observation')).keys() for group in groups[1:]])

            rollouts_all = []
            for td in rollouts:
                groups_dict_list = []
                for group in self.experiment.group_map.keys():
                    group_shared = TensorDict({
                        'observation': td.get(("next", group, 'observation')).select(*shared_keys_observation),
                        'reward': td.get(("next", group, 'reward'))
                    }, batch_size=td.batch_size)
                    groups_dict_list.append(group_shared)
                rollouts_all.append(torch.stack(groups_dict_list, dim=1))
            self._write_metrics('players', rollouts_all, self.json_writers['players'])
            del rollouts_all
            if len(roles.keys()) > 1:
                for r in roles.keys():
                    writer_key = f"players_r{r}"
                    if writer_key not in self.json_writers.keys():
                        algo_name, task_name = get_algo_and_task_name(self.experiment.task_name, self.experiment.algorithm_name, writer_key)
                        self.json_writers[writer_key] = JsonWriter(
                            folder=self.experiment.folder_name,
                            name=f"{self.experiment.name}_ssd_{writer_key}.json",
                            algorithm_name=algo_name,
                            task_name=task_name,
                            environment_name=self.experiment.environment_name,
                            seed=self.experiment.seed,
                        )

                    rollouts_role = []
                    for td in rollouts:
                        groups_dict_list = []
                        for group in roles[r]:
                            groups_dict_list.append(td.get(("next", group)))
                        rollouts_role.append(torch.stack(groups_dict_list, dim=1))

                    self._write_metrics(group, rollouts_role, self.json_writers[writer_key])



@hydra.main(version_base=None, config_path="ssd_conf", config_name="mp_config")
def hydra_experiment(cfg: DictConfig) -> None:
    """Runs an experiment loading its config from hydra.

    This function is decorated as ``@hydra.main`` and is called by running

    .. code-block:: console

       python benchmarl/run.py algorithm=mappo task=vmas/balance


    Args:
        cfg (DictConfig): the hydra config dictionary

    """
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task
    algorithm_name = hydra_choices.algorithm

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nAlgorithm: {algorithm_name}, Task: {task_name}")
    print("\nLoaded config:\n")
    print(OmegaConf.to_yaml(cfg))

    experiment = load_experiment_from_hydra(cfg, task_name=task_name, callbacks=[SSDCallback()])
    print(f'\nExperiment name: {experiment.name}\n')
    experiment.run()


if __name__ == "__main__":
    hydra_experiment()
