import jax
import jax.numpy as jnp
from typing import Optional
from nicegui import ui

from flax import struct
import numpy as np
import jax.tree_util as jtu
import jax.numpy as jnp

from housemaze import renderer
from housemaze.env import KeyboardActions
from housemaze import utils
from housemaze import env as maze

import nicewebrl
from nicewebrl import JaxWebEnv, base64_npimage, TimeStep, TimestepWrapper
from nicewebrl import Stage, EnvStage
from nicewebrl import get_logger
from nicewebrl.experiment import SimpleExperiment


logger = get_logger(__name__)

MAX_STAGE_EPISODES = 1
MAX_EPISODE_TIMESTEPS = 10000
MIN_SUCCESS_EPISODES = 1
VERBOSITY = 1


########################################
# Define actions and corresponding keys
########################################
actions = [
    KeyboardActions.right,
    KeyboardActions.down,
    KeyboardActions.left,
    KeyboardActions.up,
    KeyboardActions.done,
]
action_array = jnp.array([a.value for a in actions])
action_keys = ["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp", "d"]
action_to_name = [a.name for a in actions]


########################################
# Define mazes
########################################
maze1 = """
.............
.............
.............
.............
...#######...
...#.....#...
...#..>..#...
...#.A...#...
...#...B.#...
...#######...
.............
.............
.............
""".strip()

maze2 = """
.#.C...##....
.#..D...####.
.######......
......######.
.#.#..#......
.#.#.##..#...
##.#.#>.###.#
A..#.##..#...
.B.#.........
#####.#..####
......####.#.
.######E.#.#.
........F#...
""".strip()

########################################
# Define JaxMaze environment
########################################

def groups_to_char2key(group_set):
  chars = ["A", "B", "C", "D", "E", "F", "G", "H"]
  char2key = dict()
  for idx, char in enumerate(chars):
    i, j = idx // 2, idx % 2
    if i >= len(group_set):
      break
    char2key[char] = group_set[i, j]
  return char2key

image_data = utils.load_image_dict()


image_keys = image_data["keys"]
groups = [
    # room 1
    [image_keys.index("orange"), image_keys.index("potato")],
    # room 2
    [image_keys.index("lettuce"), image_keys.index("apple")],
    # room 3
    [image_keys.index('tomato'), image_keys.index('lettuce')],
]
groups = np.array(groups, dtype=np.int32)
task_objects = groups.reshape(-1)

# can auto-generate this from group_set
char2idx = groups_to_char2key(groups)

# shared across all tasks
task_runner = maze.TaskRunner(task_objects=task_objects)
jax_env = maze.HouseMaze(
    task_runner=task_runner,
    num_categories=len(image_data["keys"]),
)
jax_web_env = JaxWebEnv(env=jax_env, actions=action_array)

object_to_index = {key: idx for idx, key in enumerate(image_data["keys"])}

map1_init = utils.from_str(
    maze1, char_to_key=char2idx, object_to_index=object_to_index
)
map2_init = utils.from_str(
    maze2, char_to_key=char2idx, object_to_index=object_to_index
)
map_init = jtu.tree_map(lambda *v: jnp.stack(v), *(map1_init, map2_init))

example_env_params = maze.EnvParams(
    map_init=jax.tree_util.tree_map(jnp.asarray, map_init),
    time_limit=jnp.array(50),
    objects=task_objects,
)
example_env_params = jtu.tree_map(lambda x: jnp.asarray(x, dtype=jnp.int32), example_env_params)

# Call this function to pre-compile jax functions before experiment starts.
jax_web_env.precompile(dummy_env_params=example_env_params)


# jit it so fast
def housemaze_render_fn(
    timestep: maze.TimeStep, include_objects: bool = True
) -> jnp.ndarray:
  image = renderer.create_image_from_grid(
      timestep.state.grid,
      timestep.state.agent_pos,
      timestep.state.agent_dir,
      image_data,
      include_objects=include_objects,
  )
  return image
render_fn = jax.jit(housemaze_render_fn)

# precompile vmapped render fn that will vmap over all actions
vmap_render_fn = jax_web_env.precompile_vmap_render_fn(
  render_fn, example_env_params
)

rng = jax.random.PRNGKey(0)
timestep = jax_web_env.reset(rng, example_env_params)
next_timesteps = jax_web_env.next_steps(rng, timestep, example_env_params)

########################################
# Define Stages of experiment
########################################
all_stages = []


# ------------------
# Instruction stage
# ------------------
async def instruction_display_fn(stage, container):
  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)
    ui.markdown(f"## {stage.name}")
    ui.markdown(
      """
          - Press the arrow keys to move the agent
          - Press the space bar to interact with objects
          """
    )


instruction_stage = Stage(name="Instructions", display_fn=instruction_display_fn)
all_stages.append(instruction_stage)


# ------------------
# Environment stage
# ------------------
# EXAMPLE: change parameters for this specific stage
env_params = example_env_params.replace(
    time_limit=MAX_EPISODE_TIMESTEPS,
)


def make_image_html(src):
  html = f"""
  <div id="stateImageContainer" style="display: flex; justify-content: center; align-items: center;">
      <img id="stateImage" src="{src}" style="width: 100%; height: 100%; object-fit: contain;">
  </div>
  """
  return html


async def env_stage_display_fn(
  stage: EnvStage, container: ui.element, timestep: TimeStep
):
  state_image = stage.render_fn(timestep)
  state_image = base64_npimage(state_image)
  stage_state = stage.get_user_data("stage_state")

  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)
    # --------------------------------
    # tell person how many episodes completed and how many successful
    # --------------------------------
    with ui.row():
      with ui.element("div").classes("p-2 bg-blue-100"):
        ui.label(
          f"Number of successful episodes: {stage_state.nsuccesses}/{stage.min_success}"
        )
      with ui.element("div").classes("p-2 bg-green-100"):
        ui.label().bind_text_from(
          stage_state, "nepisodes", lambda n: f"Try: {n}/{stage.max_episodes}"
        )

    # --------------------------------
    # display environment
    # --------------------------------
    ui.html(make_image_html(src=state_image))


def evaluate_success_fn(timestep: TimeStep, params: Optional[struct.PyTreeNode] = None):
  """Episode finishes if person gets 5 achievements"""
  return timestep.reward.sum() > 0


environment_stage = EnvStage(
  name="Environment",
  web_env=jax_web_env,
  action_keys=action_keys,
  action_to_name=action_to_name,
  env_params=env_params,
  render_fn=render_fn,
  vmap_render_fn=vmap_render_fn,
  display_fn=env_stage_display_fn,
  evaluate_success_fn=evaluate_success_fn,
  min_success=MIN_SUCCESS_EPISODES,
  max_episodes=MAX_STAGE_EPISODES,
  verbosity=VERBOSITY,
  # add custom metadata to be stored here
  metadata=dict(
    # nothing required, just for bookkeeping
    desc="some description",
    key1="value1",
    key2="value2",
  ),
)
all_stages.append(environment_stage)

experiment = SimpleExperiment(
  stages=all_stages,
  name="JaxMaze Demo"
)