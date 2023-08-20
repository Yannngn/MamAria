import functools
import logging

logger = logging.getLogger()


def log(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"function {func.__name__} called")
        try:
            result = func(*args, **kwargs)

            logger.info(f"Object {result.__class__} was initiated")

            return result
        except Exception as exc:
            logger.exception(f"Exception raised in {func.__name__}. exception: {str(exc)}")
            raise exc

    return wrapper


def log_list(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"function {func.__name__} called")
        try:
            result = func(*args, **kwargs)

            for res in result:
                logger.info(f"Object {res.__class__} was initiated")

            return result
        except Exception as exc:
            logger.exception(f"Exception raised in {func.__name__}. exception: {str(exc)}")
            raise exc

    return wrapper
