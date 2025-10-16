from meltingpot import substrate
import cv2
from meltingpot.examples.gym import utils
from gymnasium import spaces
import time
import numpy as np
import argparse
import random

RESIZE_FACTOR = 10

def make_video_from_rgb_imgs(
    rgb_arrs, vid_path, video_name="trajectory", fps=8, format="mp4v", resize=None, frame_to_save=None
):
    """
    Create a video from a list of rgb arrays
    """
    print("Rendering video...")
    if vid_path[-1] != "/":
        vid_path += "/"
    video_path = vid_path + video_name + ".mp4"

    if resize is not None:
        width, height = resize
    else:
        frame = rgb_arrs[0]
        height, width, _ = frame.shape
        resize = width, height

    fourcc = cv2.VideoWriter_fourcc(*format)
    video = cv2.VideoWriter(video_path, fourcc, float(fps), (width, height))

    for i, image in enumerate(rgb_arrs):
        percent_done = int((i / len(rgb_arrs)) * 100)
        if percent_done % 20 == 0:
            print("\t...", percent_done, "% of frames rendered")
        # Always resize, without this line the video does not render properly.
        image = cv2.resize(image.astype(np.uint8), resize, interpolation=cv2.INTER_NEAREST)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if frame_to_save is not None and i == int(frame_to_save):
            cv2.imwrite(f'{vid_path}_{video_name}_f{i}.png', image)
        video.write(image)

    video.release()
    print("Video is created as", video_path)

def convert_spaces_tuple_to_dict(
      input_tuple: spaces.Tuple,
      remove_world_observations: bool = False,
      player_ids: list = None) -> spaces.Dict:
    """Returns spaces tuple converted to a dictionary.

    Args:
      input_tuple: tuple to convert.
      remove_world_observations: If True will remove non-player observations.
    """
    return spaces.Dict({
        agent_id: (utils.remove_world_observations_from_space(input_tuple[i])
                   if remove_world_observations else input_tuple[i])
        for i, agent_id in enumerate(player_ids)
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-e", "--env", default=None, help="environment to run a random episode")
    parser.add_argument("-n", "--num_steps", default=100, help="number of steps for the episode")
    parser.add_argument("-f", "--frame_to_save", default=None, help="frame to save as a png file")
    parser.add_argument("--human", action="store_true", help="human player or not")
    args = vars(parser.parse_args())

    substrate_name = args['env']
    env_config = substrate.get_config(substrate_name)

    env = substrate.build(substrate_name, roles=env_config.default_player_roles)
    print("Environment with", env_config.default_player_roles)
    num_players = len(env.observation_spec())

    ordered_agent_ids = [
        'player_{index}'.format(index=index)
        for index in range(num_players)
    ]

    action_space = convert_spaces_tuple_to_dict(input_tuple=utils.spec_to_space(env.action_spec()), player_ids=ordered_agent_ids)
    print("--- Action Spaces ----\n", action_space)

    frames_top = []
    frames_pov = {}
    for i in range(num_players):
        frames_pov[i] = []

    returns = {}
    for agent_id in ordered_agent_ids:
        returns[agent_id] = 0

    timestep = env.reset()

    observation = env.observation()
    world_rgb = observation[0]['WORLD.RGB']
    frames_top.append(observation[0]['WORLD.RGB'])
    for i in range(num_players):
        frames_pov[i].append(observation[i]['RGB'])

    world_h, world_w, _ = observation[0]['WORLD.RGB'].shape
    pov_h, pov_w, _ = observation[0]['RGB'].shape

    start_time = time.time()
    step = 0
    done = False
    while step < int(args['num_steps']):
        print(f"------------ Step {step} ------------")

        if args['human']:
            # Human player
            action_dict = {}
            for agent_id in ordered_agent_ids:
                aid = int(agent_id.replace('player_', ''))
                print(f"Agent {agent_id}")
                if 'POSITION' in observation[aid] and 'ORIENTATION' in observation[aid]:
                    print(f"at {observation[aid]['POSITION']} with orientation of {observation[aid]['ORIENTATION']}")

            is_exit = False
            for agent_id in ordered_agent_ids:
                aid = int(agent_id.replace('player_', ''))
                action_dict[agent_id] = int(input(f"Action for {agent_id}: "))
                if action_dict[agent_id] == 9:
                    is_exit = True

            if is_exit:
                break
        else:
            action_dict = action_space.sample()

        actions = [action_dict[agent_id] for agent_id in ordered_agent_ids]
        timestep = env.step(actions)

        rewards = {
            agent_id: timestep.reward[index]
            for index, agent_id in enumerate(ordered_agent_ids)
        }

        observation = env.observation()

        for agent_id in ordered_agent_ids:
            aid = int(agent_id.replace('player_', ''))
            print(f"Agent {agent_id}")
            if 'POSITION' in observation[aid] and 'ORIENTATION' in observation[aid]:
                print(f"at {observation[aid]['POSITION']} with orientation of {observation[aid]['ORIENTATION']}")
            if 'PLAYER_ROLE_INDEX' in observation[aid]:
                print(f"role of {int(observation[aid]['PLAYER_ROLE_INDEX'])}")
            print(f"took action {action_dict[agent_id]} and got reward {rewards[agent_id]}")

            if 'PLAYER_CALLED_GIFT' in observation[aid].keys():
                print('PLAYER_CALLED_GIFT:', observation[aid]['PLAYER_CALLED_GIFT'])

            if "SPEED" in observation[aid]:
                print(f"Agent speed: {observation[aid]['SPEED']}")

            returns[agent_id] += rewards[agent_id]

        world_rgb = observation[0]['WORLD.RGB']
        frames_top.append(observation[0]['WORLD.RGB'])
        for i in range(num_players):
            frames_pov[i].append(observation[i]['RGB'])
        step += 1

    duration = time.time() - start_time
    fps = step / duration
    print("{}: FPS: {:.2f} steps/second, duration: {}".format(substrate_name, fps, duration))

    for agent_id in ordered_agent_ids:
        print(f"Agent {agent_id}, collected return of {returns[agent_id]}")

    top_video_name = "{}".format(substrate_name)
    make_video_from_rgb_imgs(frames_top, vid_path='.', video_name=top_video_name, resize=(world_w*RESIZE_FACTOR, world_h*RESIZE_FACTOR), frame_to_save=args['frame_to_save'])

    for i in range(num_players):
        pov_video_name = "{}_agent{}_pov".format(substrate_name, i)
        make_video_from_rgb_imgs(frames_pov[i], vid_path='.', video_name=pov_video_name, resize=(pov_w*RESIZE_FACTOR, pov_h*RESIZE_FACTOR), frame_to_save=args['frame_to_save'])
