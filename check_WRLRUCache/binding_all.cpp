#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep

#include "absl/container/node_hash_map.h"

namespace nb = nanobind;

namespace xla {

namespace {

// A simple LRU cache. Not thread-safe.
// Value must be copyable and moveable. The intent is that Value is typically
// a smart-pointer type.
template <typename Key, typename Value,
          typename Hash = typename absl::node_hash_map<Key, Value>::hasher,
          typename Eq = typename absl::node_hash_map<Key, Value>::key_equal>
class LRUCache {
 private:
  struct LRUListEntry {
    LRUListEntry* next;
    LRUListEntry* prev;
  };

 public:
  // Multiple LRUCaches can share a LRU list, meaning that the capacity and
  // eviction policy is shared. The user provides an LRU list
  // to the cache constructor, and must ensure that it remains alive as long
  // as the cache does.
  class LRUList {
   public:
    explicit LRUList(int capacity) : capacity_(capacity) {
      head_.next = &head_;
      head_.prev = &head_;
    }
    ~LRUList() {
      // CHECK(head_.next == &head_);
      // CHECK(head_.prev == &head_);
    }

    LRUList(const LRUList&) = delete;
    LRUList(LRUList&&) = delete;
    LRUList& operator=(const LRUList&) = delete;
    LRUList& operator=(LRUList&&) = delete;

    int Capacity() const { return capacity_; }
    int Size() const { return size_; }

    void Clear();

   private:
    friend class LRUCache;
    int capacity_;
    int size_ = 0;

    // Root of a circular doubly-linked list of entries, in order from least
    // recently used to most recently used. An "empty" cache always contains
    // this element in the LRU list.
    LRUListEntry head_;
  };

  explicit LRUCache(LRUList* lru_list) : lru_list_(lru_list) {}
  ~LRUCache();

  LRUCache(const LRUCache&) = delete;
  LRUCache(LRUCache&&) = delete;
  LRUCache& operator=(const LRUCache&) = delete;
  LRUCache& operator=(LRUCache&&) = delete;

  // Returns the `value` associated with `key`. Creates a value with `factory`
  // and inserts it if absent.
  Value GetOrCreateIfAbsent(const Key& key,
                            const std::function<Value(const Key&)>& factory);

  void Remove(const Key& key);

  // Removes all entries from the cache.
  void Clear();

  int Size() const { return entries_.size(); }
  int Capacity() const { return lru_list_->Capacity(); }

  auto begin() const { return entries_.begin(); }
  auto end() const { return entries_.end(); }

 private:
  LRUList* lru_list_;

  struct Entry : public LRUListEntry {
    Entry() = default;

    // Pointer to the key in `entries_`. std::unordered_map<> promises
    // pointer stability for keys.
    const Key* key;
    LRUCache* container;
    std::optional<Value> value;
  };

  // We use `unordered_map` because (a) we want to guarantee pointer stability
  // for keys and values, and (b) we need exception safety so we can't use
  // absl hashtables.
  std::unordered_map<Key, Entry, Hash, Eq> entries_;
};

template <typename Key, typename Value, typename Hash, typename Eq>
void LRUCache<Key, Value, Hash, Eq>::LRUList::Clear() {
  while (head_.next != &head_) {
    static_cast<Entry*>(head_.next)->container->Clear();
  }
  size_ = 0;
}

template <typename Key, typename Value, typename Hash, typename Eq>
void LRUCache<Key, Value, Hash, Eq>::Clear() {
  for (auto& e : entries_) {
    LRUListEntry* l = &e.second;
    l->next->prev = l->prev;
    l->prev->next = l->next;
    --lru_list_->size_;
  }
  entries_.clear();
}

template <typename Key, typename Value, typename Hash, typename Eq>
LRUCache<Key, Value, Hash, Eq>::~LRUCache() {
  Clear();
}

template <typename Key, typename Value, typename Hash, typename Eq>
void LRUCache<Key, Value, Hash, Eq>::Remove(const Key& key) {
  LRUListEntry* l = &entries_[key];
  l->next->prev = l->prev;
  l->prev->next = l->next;
  --lru_list_->size_;

  entries_.erase(key);
}

template <typename Key, typename Value, typename Hash, typename Eq>
Value LRUCache<Key, Value, Hash, Eq>::GetOrCreateIfAbsent(
    const Key& key, const std::function<Value(const Key&)>& factory) {
  auto [it, inserted] = entries_.try_emplace(key);
  Entry& entry = it->second;
  if (inserted) {
    entry.key = &it->first;
    entry.value = factory(*entry.key);
    ++lru_list_->size_;
  } else {
    // Removes the entry from the LRU list, in preparation for adding it
    // to the back of the list.
    entry.prev->next = entry.next;
    entry.next->prev = entry.prev;
  }
  // (Re-)adds entry to the back of the LRU list. Since it is now the
  // most recently used element, it goes at the back.
  LRUListEntry& lru_head = lru_list_->head_;
  entry.container = this;
  entry.prev = lru_head.prev;
  entry.next = &lru_head;
  lru_head.prev->next = &entry;
  lru_head.prev = &entry;

  Value v = *entry.value;

  // Evict an LRU entry if we are over capacity.
  if (lru_list_->size_ > lru_list_->capacity_) {
    Entry* to_remove = static_cast<Entry*>(lru_head.next);
    to_remove->next->prev = &lru_head;
    lru_head.next = to_remove->next;
    // Extract instead of erase in case the kv pair contains python objects
    // whose destruction could call back into this code. Extract causes the
    // dtor to be delayed until the kv pair is fully removed from the map.
    to_remove->container->entries_.extract(*to_remove->key);
    --lru_list_->size_;
  }
  return v;
}

} // xla_pjrt


namespace {

// Minimal wrapper to expose a nb::dict_iterator's value as something
// hashable with Abseil.
class HashablePyDictEntry {
 public:
  explicit HashablePyDictEntry(std::pair<nb::handle, nb::handle> entry)
      : entry_(entry) {}

  template <typename H>
  friend H AbslHashValue(H h, const HashablePyDictEntry& v) {
    return H::combine(std::move(h), nb::hash(v.entry_.first),
                      nb::hash(v.entry_.second));
  }

  std::pair<nb::handle, nb::handle> entry_;
};

// Similarly, a minimalist adaptor around the nb::detail::dict_iterator
// itself. Note that the iterator "is" also a Value. Does not meet the full
// standard iterator requirements, only enough to support H::combine_unordered.
class HashablePyDictIter {
 public:
  using iterator_category = std::input_iterator_tag;

  explicit HashablePyDictIter(nb::detail::dict_iterator& iter) : iter_(iter) {}

  // Minimal set of iterator operations.
  HashablePyDictEntry operator*() const { return HashablePyDictEntry(*iter_); }
  bool operator!=(const HashablePyDictIter& rhs) const {
    return iter_ != rhs.iter_;
  }
  void operator++() { ++iter_; }

 private:
  nb::detail::dict_iterator& iter_;
};

}  // namespace


class WeakrefLRUCache : public std::enable_shared_from_this<WeakrefLRUCache> {
 public:
  struct Key {
    nb::object context;
    nb::args args;
    nb::kwargs kwargs;

    bool operator==(const Key& other) const {
      return context.equal(other.context) && args.equal(other.args) &&
             kwargs.equal(other.kwargs);
    }

    template <typename H>
    friend H AbslHashValue(H h, const Key& key) {
      // Note: Despite the fact this is an ABSL hash function, it's safe to call
      // functions that may throw exceptions such as nb::hash(), because it is
      // used by an LRUCache, which uses a std::unordered_map, which is
      // exception-safe.
      h = H::combine(std::move(h), nb::hash(key.context), nb::hash(key.args));
      nb::detail::dict_iterator begin = key.kwargs.begin();
      nb::detail::dict_iterator end = key.kwargs.end();
      h = H::combine_unordered(std::move(h), HashablePyDictIter(begin),
                               HashablePyDictIter(end));
      h = H::combine(std::move(h), key.kwargs.size());
      return h;
    }
  };

  struct CacheEntry {
    bool has_result = false;
    nb::object result;
    absl::Notification completed;
    std::thread::id thread_id = std::this_thread::get_id();
  };

  struct CacheInfo {
    int64_t hits;
    int64_t misses;
    int64_t maxsize;
    int64_t currsize;
  };

  struct WeakrefCacheKey {
    nb::handle object;
    size_t cached_hash;
  };

  using Cache = LRUCache<Key, std::shared_ptr<CacheEntry>>;

  struct WeakrefCacheValue {
    std::optional<nb::weakref> weakref;
    std::shared_ptr<Cache> cache;
  };

  struct WeakrefKeyHash {
    size_t operator()(const WeakrefCacheKey& v) const { return v.cached_hash; }
  };

  struct WeakrefKeyEq {
    bool operator()(const WeakrefCacheKey& lhs,
                    const WeakrefCacheKey& rhs) const {
      return lhs.object.equal(rhs.object);
    }
  };

  WeakrefLRUCache(nb::callable cache_context_fn, nb::callable fn,
                  int64_t maxsize)
      : cache_context_fn_(cache_context_fn), fn_(fn), lru_list_(maxsize) {}

  std::shared_ptr<Cache> GetCache(WeakrefCacheKey key) {
#ifdef NB_FREE_THREADED
    mu_.Lock();
    absl::Cleanup unlock = [this]() ABSL_UNLOCK_FUNCTION(mu_) { mu_.Unlock(); };
#endif

    auto [it, inserted] = entries_.emplace(key, WeakrefCacheValue());
    if (!inserted) {
      return it->second.cache;
    }

    auto& value = it->second;

    value.cache = std::make_shared<Cache>(&lru_list_);
    auto weakref_gc_callback = nb::cpp_function(
        [this_weak = weak_from_this(), key](nb::handle weakref) {
          auto cache = this_weak.lock();
          if (cache == nullptr) {
            return;
          }
#ifdef NB_FREE_THREADED
          cache->mu_.Lock();
#endif
          auto it = cache->entries_.find(key);
          if (it != cache->entries_.end()) {
            // Create temp-var to avoid re-entrant erase.
            auto tmp = std::move(it->second);
            cache->entries_.erase(it);
          }
#ifdef NB_FREE_THREADED
          cache->mu_.Unlock();
#endif
        });
    PyObject* ref =
        PyWeakref_NewRef(key.object.ptr(), weakref_gc_callback.ptr());
    if (!ref) {
      entries_.erase(it);
      throw nb::python_error();
    }
    value.weakref = nb::steal<nb::weakref>(ref);
    return value.cache;
  }

  nb::object Call(nb::object weakref_key, nb::args args,
                  nb::kwargs kwargs) ABSL_NO_THREAD_SAFETY_ANALYSIS {
#ifdef NB_FREE_THREADED
    mu_.Lock();
#endif
    auto wrcachekey = WeakrefCacheKey{
        weakref_key, static_cast<size_t>(nb::hash(weakref_key))};
#ifdef NB_FREE_THREADED
    mu_.Unlock();
#endif
    std::shared_ptr<Cache> cache_ptr = GetCache(wrcachekey);
#ifdef NB_FREE_THREADED
    mu_.Lock();
    absl::Cleanup unlock = [this]() ABSL_UNLOCK_FUNCTION(mu_) { mu_.Unlock(); };
#endif
    Cache& cache = *cache_ptr;

    nb::object context = cache_context_fn_();
    ++total_queries_;

    bool inserted = false;
    std::shared_ptr<CacheEntry> entry;
#ifndef NB_FREE_THREADED
    {
      // Because the gil can be released during cache insertion, this forces
      // the lock order to be mu_ then gil so we must release the gil first.
      nb::gil_scoped_release release;
      // Acquire a mutex to avoid problems where the gil is released during
      // cache insertion and then a second thread invalidates the cache order.
      mu_.Lock();
    }
#endif
    {
#ifndef NB_FREE_THREADED
      // GetOrCreateIfAbsent calls into Python hash and equality functions,
      // which may throw exceptions. The use of absl::Cleanup ensures mu_ is
      // released if that happens.
      absl::Cleanup unlock = [this]()
                                 ABSL_UNLOCK_FUNCTION(mu_) { mu_.Unlock(); };
#endif
      Key key{context, args, kwargs};
      entry = cache.GetOrCreateIfAbsent(key, [&inserted](const Key& key) {
        inserted = true;
        return std::make_shared<CacheEntry>();
      });
    }
    if (!entry->completed.HasBeenNotified()) {
      if (inserted) {
        ++misses_;
        absl::Cleanup notify = [&] { entry->completed.Notify(); };
        entry->result = fn_(weakref_key, *args, **kwargs);
        entry->has_result = true;
      } else {
        if (entry->thread_id == std::this_thread::get_id()) {
          auto error_string =
              absl::StrCat("Recursively calling ",
                           nb::cast<std::string>(nb::repr(weakref_key)),
                           nb::cast<std::string>(nb::repr(args)));
          PyErr_SetString(PyExc_RecursionError, error_string.c_str());
          throw nb::python_error();
        }
        nb::gil_scoped_release release;
        entry->completed.WaitForNotification();
      }
    }

    if (entry->has_result) {
      return entry->result;
    } else {
      ++misses_;
      return fn_(weakref_key, *args, **kwargs);
    }
  }
  std::vector<nb::object> GetKeys() {
    std::vector<nb::object> results;
    mu_.Lock();
    for (const auto& wr_entry : entries_) {
      for (const auto& rest : *wr_entry.second.cache) {
        nb::tuple result =
            nb::make_tuple(*wr_entry.second.weakref, rest.first.context,
                           rest.first.args, rest.first.kwargs);
        results.push_back(std::move(result));
      }
    }
    mu_.Unlock();
    return results;
  }
  CacheInfo GetCacheInfo() {
#ifdef NB_FREE_THREADED
    mu_.Lock();
    absl::Cleanup unlock = [this]() ABSL_UNLOCK_FUNCTION(mu_) { mu_.Unlock(); };
#endif
    CacheInfo result;
    result.hits = total_queries_ - misses_;
    result.misses = misses_;
    result.maxsize = lru_list_.Capacity();
    result.currsize = lru_list_.Size();
    return result;
  }
  void Clear() {
#ifdef NB_FREE_THREADED
    mu_.Lock();
    absl::Cleanup unlock = [this]() ABSL_UNLOCK_FUNCTION(mu_) { mu_.Unlock(); };
#endif
    total_queries_ = misses_ = 0;
    std::vector<std::shared_ptr<Cache>> deferred_deletes;
    deferred_deletes.reserve(entries_.size());
    for (auto& entry : entries_) {
      deferred_deletes.push_back(std::move(entry.second.cache));
    }
    entries_.clear();
    deferred_deletes.clear();
  }

  nb::callable cache_context_fn_;
  nb::callable fn_;
  Cache::LRUList lru_list_;
  std::unordered_map<WeakrefCacheKey, WeakrefCacheValue, WeakrefKeyHash,
                     WeakrefKeyEq>
      entries_;
  int64_t misses_ = 0;
  int64_t total_queries_ = 0;
  absl::Mutex mu_;
};

void BuildWeakrefLRUCacheAPI(nb::module_& m) {
  auto weakref_lru_cache =
      nb::class_<WeakrefLRUCache>(m, "WeakrefLRUCache",
                                  nb::is_weak_referenceable())
          .def("__call__", &WeakrefLRUCache::Call)
          .def("cache_keys", &WeakrefLRUCache::GetKeys)
          .def("cache_info", &WeakrefLRUCache::GetCacheInfo)
          .def("cache_clear", &WeakrefLRUCache::Clear);
  nb::class_<WeakrefLRUCache::CacheInfo>(weakref_lru_cache,
                                         "WeakrefLRUCacheInfo")
      .def_ro("hits", &WeakrefLRUCache::CacheInfo::hits)
      .def_ro("misses", &WeakrefLRUCache::CacheInfo::misses)
      .def_ro("maxsize", &WeakrefLRUCache::CacheInfo::maxsize)
      .def_ro("currsize", &WeakrefLRUCache::CacheInfo::currsize)
      .def("__repr__", [](WeakrefLRUCache::CacheInfo& info) {
        return absl::StrCat(
            "WeakrefLRUCache(hits=", info.hits, ", misses=", info.misses,
            ", maxsize=", info.maxsize, ", currsize=", info.currsize, ")");
      });
  m.def(
      "weakref_lru_cache",
      [](nb::callable cache_context_fn, nb::callable fn, int64_t maxsize) {
        return std::make_shared<WeakrefLRUCache>(cache_context_fn, fn, maxsize);
      },
      nb::arg("cache_context_fn"), nb::arg("fn"), nb::arg("maxsize") = 2048);
}

}  // namespace jax

NB_MODULE(wrlru_cache_ext, m) {
    xla::BuildWeakrefLRUCacheAPI(m);
}
