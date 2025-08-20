import os.path
import asyncio
from asyncio import Lock
from nicegui import app, ui
from fastapi import Request
from tortoise import Tortoise
import jax
import httpx
import config
import nicewebrl
from nicewebrl.logging import setup_logging, get_logger
from nicewebrl.utils import wait_for_button_or_keypress
from nicewebrl import stages, TimeStep

import nicewebrl
import os
from importlib.util import find_spec
import shutil
import asyncio
import json
import time
from nicewebrl.logging import get_logger
from nicegui import app

# Google Cloud Storage functionality (copied from xland-LLM-assistant)
from dotenv import load_dotenv
from google.auth.exceptions import TransportError
from google.cloud import storage
from google.cloud.exceptions import exceptions as gcs_exceptions

from experiment_structure import experiment

load_dotenv()
logger = get_logger(__name__)

GOOGLE_CREDENTIALS = "./google-cloud-key.json"
# Google Cloud Storage
BUCKET_NAME = "jaxmaze"

# Data
DATA_DIR = "./data"
DATABASE_FILE = "db.sqlite"

_user_locks = {}
async def save_file_to_gcs(local_filename, blob_filename, bucket_name: str):
  try:
    storage_client = storage.Client.from_service_account_json(GOOGLE_CREDENTIALS)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_filename)

    # Run the blocking upload in a thread pool
    await asyncio.to_thread(blob.upload_from_filename, local_filename)

    logger.info(f"Saved {blob_filename} in bucket {bucket.name}")
    return True  # Successfully saved
  except Exception as e:
    logger.info(f"Unexpected error: {e}")
    logger.info("Skipping GCS upload")

  return False  # Failed to save


async def save_to_gcs_with_retries(
  files_to_save, max_retries=5, retry_delay=5, bucket_name: str = ""
):
  """Save multiple files to Google Cloud Storage with retry logic.

  Args:
      files_to_save: List of filenames
      max_retries: Number of retry attempts
      retry_delay: Seconds to wait between retries
      bucket_name: Name of the GCS bucket

  Returns:
      bool: True if all files were saved successfully, False otherwise
  """
  assert bucket_name != "", "Bucket name is required"
  for attempt in range(max_retries):
    try:
      # Try to save all files
      for local_file in files_to_save:
        saved = await save_file_to_gcs(
          local_filename=local_file, blob_filename=local_file, bucket_name=bucket_name
        )
        if not saved:
          raise Exception(f"Failed to save {local_file}")

      logger.info(f"Successfully saved data to GCS on attempt {attempt + 1}")
      return True

    except Exception as e:
      if attempt < max_retries - 1:
        logger.info(f"Error saving to GCS: {e}. Retrying in {retry_delay} seconds...")
        await asyncio.sleep(retry_delay)
      else:
        logger.info(f"Failed to save to GCS after {max_retries} attempts: {e}")
        return False


async def finish_experiment(feedback=None, **kwargs):
  """Function to save data to Google Cloud Storage on experiment termination.
  
  This function requires:
  - google-cloud-key.json file in the same directory
  - BUCKET_NAME environment variable set
  - google-cloud-storage package installed
  """
  bucket_name = os.getenv("BUCKET_NAME")
  if not GOOGLE_CREDENTIALS or not os.path.exists(GOOGLE_CREDENTIALS):
    logger.warning("No Google credentials found, skipping save")
    return
  
  if not bucket_name:
    logger.warning("No BUCKET_NAME environment variable found, skipping save")
    return

  user_data_file = nicewebrl.user_data_file()
  user_metadata_file = nicewebrl.user_metadata_file()

  # --------------------------------
  # save user data to final line of file
  # --------------------------------
  user_storage = nicewebrl.make_serializable(dict(app.storage.user))
  metadata = dict(
    finished=True,
    feedback=feedback,
    user_storage=user_storage,
    **kwargs,
  )
  
  with open(user_metadata_file, "w") as f:
    json.dump(metadata, f)

  files_to_save = [user_data_file, user_metadata_file]
  logger.info(f"Saving to bucket: {bucket_name}")
  await save_to_gcs_with_retries(
    files_to_save,
    max_retries=5,
    bucket_name=bucket_name,
  )

  # Try to delete local files after successful upload
  from nicewebrl.stages import StageStateModel

  logger.info(f"Deleting data for user {app.storage.browser['id']}")
  await StageStateModel.filter(session_id=app.storage.browser["id"]).delete()
  logger.info(
    f"Successfully deleted stage information for user {app.storage.browser['id']}"
  )
  for local_file in files_to_save:
    try:
      os.remove(local_file)
      logger.info(f"Successfully deleted local file: {local_file}")
    except Exception as e:
      logger.warning(f"Failed to delete local file {local_file}: {str(e)}")


def get_user_lock():
  user_seed = app.storage.user["seed"]
  if user_seed not in _user_locks:
    _user_locks[user_seed] = Lock()
  return _user_locks[user_seed]


async def global_handle_key_press(e, container):
  logger.info("global_handle_key_press")
  if experiment.finished():
    logger.info("Experiment finished")
    return

  stage = await experiment.get_stage()
  if stage.get_user_data("finished", False):
    return

  await stage.handle_key_press(e, container)
  local_handle_key_press = stage.get_user_data("local_handle_key_press")
  if local_handle_key_press is not None:
    await local_handle_key_press()


setup_logging(DATA_DIR, nicegui_storage_user_key="seed")
logger = get_logger("main")

if not os.path.exists(DATA_DIR):
  os.mkdir(DATA_DIR)


async def init_db() -> None:
  await Tortoise.init(
      db_url=f"sqlite://{DATA_DIR}/{DATABASE_FILE}",
      modules={"models": ["nicewebrl.stages"]},
  )
  await Tortoise.generate_schemas()


async def close_db() -> None:
  await Tortoise.close_connections()


app.on_startup(init_db)
app.on_shutdown(close_db)


#####################################
# Consent Form and demographic info
#####################################


async def make_consent_form(container):
  consent_given = asyncio.Event()
  with container:
    ui.markdown("## Consent Form")
    with open("consent.md", "r") as consent_file:
      consent_text = consent_file.read()
    ui.markdown(consent_text)

    def on_change():
      print("on_change")
      consent_given.set()

    ui.checkbox("I agree to participate.", on_change=on_change)
  print("waiting for consent")
  await consent_given.wait()


async def collect_demographic_info(container):
  # Create a markdown title for the section
  nicewebrl.clear_element(container)
  collected_demographic_info_event = asyncio.Event()
  with container:
    ui.markdown("## Demographic Info")
    ui.markdown("Please fill out the following information.")

    with ui.column():
      with ui.column():
        ui.label("Biological Sex")
        sex_input = ui.radio(["Male", "Female"], value="Male").props("inline")

      # Collect age with a textbox input
      age_input = ui.input("Age")

    # Button to submit and store the data
    async def submit():
      age = age_input.value
      sex = sex_input.value

      # Validation for age input
      if not age.isdigit() or not (0 < int(age) < 100):
        ui.notify("Please enter a valid age between 1 and 99.", type="warning")
        return
      app.storage.user["age"] = int(age)
      app.storage.user["sex"] = sex
      logger.info(f"age: {int(age)}, sex: {sex}")
      collected_demographic_info_event.set()

    button = ui.button("Submit", on_click=submit)
    await button.clicked()


async def start_experiment(meta_container, stage_container):
  # ========================================
  # Consent form and demographic info
  # ========================================
  if not (app.storage.user.get("experiment_started", False)):
    await make_consent_form(stage_container)
    await collect_demographic_info(stage_container)
    app.storage.user["experiment_started"] = True

  # ========================================
  # Force fullscreen
  # ========================================
  # ui.run_javascript("window.require_fullscreen = true")

  # ========================================
  # Register global key press handler
  # ========================================
  ui.on("key_pressed", lambda e: global_handle_key_press(e, stage_container))

  # ========================================
  # Start experiment
  # ========================================
  logger.info("Starting experiment")

  while not experiment.finished():
    stage = await experiment.get_stage()
    await run_stage(stage, stage_container)
    await stage.finish_saving_user_data()
    await experiment.advance_stage()

  await finish_experiment(meta_container)


async def finish_experiment(container):
  nicewebrl.clear_element(container)
  with container:
    ui.markdown("# Experiment over")

  #########################
  # Save data
  #########################
  async def submit(feedback):
    app.storage.user["experiment_finished"] = True
    status_container = None
    with container:
      nicewebrl.clear_element(container)
      ui.markdown(
          "## Your data is being saved. Please do not close or refresh the page."
      )
      status_container = ui.markdown("Saving local files...")

    try:
      # Create a task for the save operation with a timeout
      save_task = asyncio.create_task(save_data(feedback=feedback))
      start_time = time.time()

      # Update status every 2 seconds while waiting for save
      while not save_task.done():
        elapsed_seconds = int(time.time() - start_time)
        status_container.content = (
            f"Still saving... ({elapsed_seconds}s elapsed). This may take 5-10 minutes."
        )
        try:
          # Wait for either task completion or timeout
          await asyncio.wait_for(asyncio.shield(save_task), timeout=2.0)
        except asyncio.TimeoutError:
          # This is expected - we use timeout to update status
          continue
        except Exception as e:
          logger.error(f"Error during save: {e}")
          status_container.content = (
              "⚠️ Error saving data. Please contact the experimenter."
          )
          raise

      # If we get here, save was successful
      elapsed_seconds = int(time.time() - start_time)
      status_container.content = (
          f"✅ Save complete in {elapsed_seconds}s! Moving to next screen..."
      )
      app.storage.user["data_saved"] = True

    except Exception as e:
      logger.error(f"Save failed: {e}")
      status_container.content = "⚠️ Error saving data. Please contact the experimenter."
      raise

  app.storage.user["data_saved"] = app.storage.user.get("data_saved", False)
  if not app.storage.user["data_saved"]:
    with container:
      nicewebrl.clear_element(container)
      ui.markdown(
          "Please provide feedback on the experiment here. For example, please describe if anything went wrong or if you have any suggestions for the experiment."
      )
      text = ui.textarea().style("width: 80%;")  # Set width to 80% of the container
      button = ui.button("Submit")
      await button.clicked()
      await submit(text.value)

  #########################
  # Final screen
  #########################
  with container:
    nicewebrl.clear_element(container)
    ui.markdown("# Experiment over")
    ui.markdown("## Data saved")
    ui.markdown(
        "### Please record the following code which you will need to provide for compensation"
    )
    ui.markdown("### 'carvalho.assistants 3'")
    ui.markdown("#### You may close the browser")


async def save_data(feedback=None, **kwargs):
  if not GOOGLE_CREDENTIALS:
    logger.warning("No Google credentials found, skipping save")
    return

  user_data_file = nicewebrl.user_data_file()
  user_metadata_file = nicewebrl.user_metadata_file()

  # --------------------------------
  # save user data to final line of file
  # --------------------------------
  user_storage = nicewebrl.make_serializable(dict(app.storage.user))
  metadata = dict(
      finished=True,
      feedback=feedback,
      user_storage=user_storage,
      **kwargs,
  )

  with open(user_metadata_file, "w") as f:
    json.dump(metadata, f)

  files_to_save = [user_data_file, user_metadata_file]
  logger.info(f"Saving to bucket: {config.BUCKET_NAME}")
  await save_to_gcs_with_retries(
      files_to_save,
      max_retries=5,
      bucket_name=config.BUCKET_NAME,
  )

  # Try to delete local files after successful upload
  from nicewebrl.stages import StageStateModel

  logger.info(f"Deleting data for user {app.storage.browser['id']}")
  await StageStateModel.filter(session_id=app.storage.browser["id"]).delete()
  logger.info(
      f"Successfully deleted stage inforation for user {app.storage.browser['id']}"
  )
  for local_file in files_to_save:
    try:
      os.remove(local_file)
      logger.info(f"Successfully deleted local file: {local_file}")
    except Exception as e:
      logger.warning(f"Failed to delete local file {local_file}: {str(e)}")


async def run_stage(stage, container):
  stage_over_event = asyncio.Event()

  async def local_handle_key_press():
    async with get_user_lock():
      if stage.get_user_data("finished", False):
        logger.info(f"Finished {stage.name} via key press")
        stage_over_event.set()

  await stage.set_user_data(local_handle_key_press=local_handle_key_press)

  async def handle_button_press():
    if stage.get_user_data("finished", False):
      return
    await stage.handle_button_press(container)
    async with get_user_lock():
      if stage.get_user_data("finished", False):
        logger.info(f"Finished {stage.name} via button press")
        stage_over_event.set()

  with container.style("align-items: center;"):
    await stage.activate(container)

  if stage.get_user_data("finished", False):
    logger.info(f"Finished {stage.name} immediately after activation")
    stage_over_event.set()

  if stage.next_button:
    with container:
      button = ui.button("Next page")
      await wait_for_button_or_keypress(button)
      await handle_button_press()

  await stage_over_event.wait()


async def check_if_over(container, episode_limit=60):
  minutes_passed = nicewebrl.get_user_session_minutes()
  minutes_passed = app.storage.user["session_duration"]
  if minutes_passed > episode_limit:
    pass


@ui.page("/")
async def index(request: Request):
  nicewebrl.initialize_user(request=request)
  await experiment.initialize()

  model_list = ["gemini", "claude", "chatgpt"]
  # Initialize random model selection if not already set
  if "selected_model" not in app.storage.user:
    rng = nicewebrl.new_rng()
    idx = jax.random.randint(rng, (), 0, len(model_list))
    app.storage.user["selected_model"] = model_list[int(idx)]

  basic_javascript_file = nicewebrl.basic_javascript_file()
  with open(basic_javascript_file) as f:
    ui.add_body_html("<script>" + f.read() + "</script>")

  card = (
      ui.card(align_items=["center"])
      .classes("fixed-center")
      .style(
          "width: 80vw;"  # Set width to 90% of viewport width
          "max-height: 90vh;"  # Keep the same max height
          "overflow: auto;"
          "display: flex;"
          "flex-direction: column;"
          "justify-content: flex-start;"
          "align-items: center;"
          "padding: 1rem;"
      )
  )
  with card:
    meta_container = ui.column()
    with meta_container.style("align-items: center;"):
      display_container = ui.row()
      with display_container.style("align-items: center;"):
        stage_container = ui.column()
        ui.timer(
            interval=10,
            callback=lambda: check_if_over(
                episode_limit=200, container=stage_container),
        )
      footer_container = ui.row()
    with meta_container.style("align-items: center;"):
      await footer(footer_container)
      with display_container.style("align-items: center;"):
        await start_experiment(display_container, stage_container)


async def footer(footer_container):
  """Add user information and progress bar to the footer"""
  with footer_container:
    with ui.row():

      ui.label().bind_text_from(app.storage.user,
                                "user_id", lambda v: f"user id: {v}.")
      ui.label()

      def text_display(v):
        stage_idx = max(experiment.num_stages, int(v) + 1)
        return f"stage: {stage_idx}/{experiment.num_stages}."

      ui.label().bind_text_from(app.storage.user, "stage_idx", text_display)
      ui.label()
      ui.label()
      ui.label().bind_text_from(
          app.storage.user, "session_duration", lambda v: f"minutes passed: {int(v)}."
      )

    ui.linear_progress(
        value=nicewebrl.get_progress()).bind_value_from(app.storage.user, "stage_progress")

    ui.button(
        "Toggle fullscreen",
        icon="fullscreen",
        on_click=nicewebrl.utils.toggle_fullscreen,
    ).props("flat")

ui.run(
    storage_secret="private key to secure the browser session cookie",
    reload="FLY_ALLOC_ID" not in os.environ,
    title="Minigrid Web App",
    port=8081,
)
