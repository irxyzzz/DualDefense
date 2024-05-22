import time

import pytest


@pytest.fixture(scope="session", autouse=True)
def timer_session_scope():
    start = time.time()
    print(
        "\nstart: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start)))
    )

    yield

    finished = time.time()
    print(
        "finished: {}".format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(finished))
        )
    )
    print("Total time cost: {:.3f}s".format(finished - start))


@pytest.fixture(autouse=True)
def timer_function_scope():
    start = time.time()
    yield
    print(" Time cost: {:.3f}s".format(time.time() - start))
