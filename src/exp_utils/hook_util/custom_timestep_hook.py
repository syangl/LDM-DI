from contextlib import contextmanager

@contextmanager
def custom_timesteps_hook(scheduler, custom_timesteps):
    """
    Hook to replace scheduler.timesteps with custom timesteps.
    """
    # original `set_timesteps` Method
    original_set_timesteps = scheduler.set_timesteps

    def hooked_set_timesteps(*args, **kwargs):
        # call original method
        original_set_timesteps(*args, **kwargs)
        # replace timesteps
        scheduler.timesteps = custom_timesteps.to(scheduler.timesteps.device)

    # install hook
    scheduler.set_timesteps = hooked_set_timesteps

    try:
        yield scheduler
    finally:
        scheduler.set_timesteps = original_set_timesteps
