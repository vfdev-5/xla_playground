
import time

import threading
import concurrent.futures
import functools
from typing import Optional
import weakref
import unittest

from wrlru_cache_ext import weakref_lru_cache


def multi_threaded(*, num_workers: int, skip_tests: Optional[list[str]] = None):
  """Decorator that runs a test in a multi-threaded environment."""

  def decorator(test_cls):
    for name, test_fn in test_cls.__dict__.copy().items():
      if not (name.startswith("test") and callable(test_fn)):
        continue

      if skip_tests is not None:
        if any(test_name in name for test_name in skip_tests):
          continue

      @functools.wraps(test_fn)  # pylint: disable=cell-var-from-loop
      def multi_threaded_test_fn(*args, __test_fn__=test_fn, **kwargs):

        barrier = threading.Barrier(num_workers)

        def closure():
          barrier.wait()
          __test_fn__(*args, **kwargs)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers
        ) as executor:
          futures = []
          for _ in range(num_workers):
            futures.append(executor.submit(closure))
          # We should call future.result() to re-raise an exception if test has
          # failed
          list(f.result() for f in futures)

      setattr(test_cls, f"{name}_multi_threaded", multi_threaded_test_fn)

    return test_cls

  return decorator


@multi_threaded(num_workers=5, skip_tests=["testMultiThreaded", "test2MultiThreaded", "test3MultiThreaded"])
class WeakrefLRUCacheTest(unittest.TestCase):

  def test3MultiThreaded(self):

    num_workers = 3
    barrier = threading.Barrier(num_workers)
    cache = weakref_lru_cache(lambda: None, lambda x, y: y, 2048)

    class WRKey:
      pass

    def worker_add_to_cache():
        barrier.wait()
        wrkey = WRKey()
        for i in range(10):
            cache(wrkey, i)

    def worker_clean_cache():
        barrier.wait()
        for i in range(10):
            cache.cache_clear()

    workers = [
        threading.Thread(target=worker_add_to_cache) for _ in range(num_workers - 1)
    ] + [
        threading.Thread(target=worker_clean_cache)
    ]

    for t in workers:
        t.start()

    for t in workers:
        t.join()

  def test2MultiThreaded(self):

    num_workers = 3
    barrier = threading.Barrier(num_workers)
    cache = weakref_lru_cache(lambda: None, lambda x, y: y, 2048)

    class WRKey:
      pass

    def worker_add_to_cache():
        barrier.wait()
        wrkey = WRKey()
        for i in range(10):
            cache(wrkey, i)

    workers = [
        threading.Thread(target=worker_add_to_cache) for _ in range(num_workers)
    ]

    for t in workers:
        t.start()

    for t in workers:
        t.join()

  def testMultiThreaded(self):
    insert_evs = [threading.Event() for _ in range(2)]
    insert_evs_i = 0

    class WRKey:
      pass

    class ClashingKey:

      def __eq__(self, other):
        return False

      def __hash__(self):
        return 333  # induce maximal caching problems.

    class GilReleasingCacheKey:

      def __eq__(self, other):
        nonlocal insert_evs_i
        if isinstance(other, GilReleasingCacheKey) and insert_evs_i < len(
            insert_evs
        ):
          insert_evs[insert_evs_i].set()
          insert_evs_i += 1
          time.sleep(0.01)
        return False

      def __hash__(self):
        return 333  # induce maximal caching problems.

    def CacheFn(obj, gil_releasing_cache_key):
      del obj
      del gil_releasing_cache_key
      return None

    cache = weakref_lru_cache(lambda: None, CacheFn, 2048)

    wrkey = WRKey()

    def Body():
      for insert_ev in insert_evs:
        insert_ev.wait()
        for _ in range(20):
          cache(wrkey, ClashingKey())

    t = threading.Thread(target=Body)
    t.start()
    for _ in range(3):
      cache(wrkey, GilReleasingCacheKey())
    t.join()

  def testKwargsDictOrder(self):
    miss_id = 0

    class WRKey:
      pass

    def CacheFn(obj, kwkey1, kwkey2):
      del obj, kwkey1, kwkey2
      nonlocal miss_id
      miss_id += 1
      return miss_id

    cache = weakref_lru_cache(lambda: None, CacheFn, 4)

    wrkey = WRKey()

    self.assertEqual(cache(wrkey, kwkey1="a", kwkey2="b"), 1)
    self.assertEqual(cache(wrkey, kwkey1="b", kwkey2="a"), 2)
    self.assertEqual(cache(wrkey, kwkey2="b", kwkey1="a"), 1)

  def testGetKeys(self):
    def CacheFn(obj, arg):
      del obj
      return arg + "extra"

    cache = weakref_lru_cache(lambda: None, CacheFn, 4)

    class WRKey:
      pass

    wrkey = WRKey()

    self.assertTrue(len(cache.cache_keys()) == 0)
    cache(wrkey, "arg1")
    cache(wrkey, "arg2")
    self.assertTrue(len(cache.cache_keys()) == 2)

  def testNonWeakreferenceableKey(self):
    class NonWRKey:
      __slots__ = ()

    non_wr_key = NonWRKey()
    with self.assertRaises(TypeError):
      weakref.ref(non_wr_key)

    cache = weakref_lru_cache(lambda: None, lambda x: 2048)
    for _ in range(100):
      with self.assertRaises(TypeError):
        cache(non_wr_key)

  def testCrashingKey(self):
    class WRKey:
      pass

    class CrashingKey:
      # A key that raises exceptions if eq or hash is called.

      def __eq__(self, other):
        raise ValueError("eq")

      def __hash__(self):
        raise ValueError("hash")

    cache = weakref_lru_cache(lambda: None, lambda x, y: y, 2048)
    wrkey = WRKey()
    with self.assertRaises(ValueError):
      for _ in range(100):
        cache(wrkey, CrashingKey())

  def testPrintingStats(self):
    class WRKey:
      pass

    cache = weakref_lru_cache(lambda: None, lambda x, y: y, 2048)
    wrkey = WRKey()
    for i in range(10):
      cache(wrkey, i)
    for i in range(5):
      cache(wrkey, i)

    self.assertEqual(
        repr(cache.cache_info()),
        "WeakrefLRUCache(hits=5, misses=10, maxsize=2048, currsize=10)",
    )


if __name__ == "__main__":
  unittest.main()
