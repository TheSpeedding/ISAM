#define AUTHOR_NAME "Lukas Riedel"

#ifndef isam_hpp
#define isam_hpp

#include <vector>
#include <stdexcept>
#include <memory>
#include <type_traits>
#include <map>
#include <algorithm>
#include "block_provider.hpp"

#define STL_BINARY_SEARCH true
#define FLUSH_OVERFLOW_IN_ITERATOR_CTOR true

// Passed 15000 automatically generated unit tests.

// You can use custom comparer for the isam. Tested with std::less and std::greater, it works like a charm. Custom allocators were not implemented.

// I think it is not a good idea to store pointer to the next block in the block itself. Suppose blocks are stored in the hard-drive. Each loading causes that the block is loaded into another address in memory.
// Storing ID of the next block is not a solution, too. Suppose we have some blocks in the hard-drive and we want to load them into the isam. The blocks may not be loaded in-order, which causes invalidation, too.
// Yeah, we could store block ID along with ID of the next block, but then we would waste 2 * sizeof(size_t) memory.

namespace isam
{
	namespace impl
	{
		template <typename Key, typename T, typename Comparer>
		class block_viewer {
			// Block contains: current size (size_t), capacity (size_t), raw data (std::pair<Key, T>). 			
			// Overhead per one block_viewer x86 (x64): 28B (56B). 
			// Overhead in memory allocated by malloc (raw block data, without data fields in this object) x86 (x64): 8B, (16B).
		private:
			union {
				// We need block size just until the first loading. Then, it will be written into the block and we will need block id instead. 
				// How to find out which of the variables is valid? If "ptr_ == nullptr", then block_size_ is valid. We will save sizeof(size_t) memory :-)

				size_t block_id_; // Block id. 

				const size_t block_size_; // Size of the block.
			};

			size_t counter_; // Indicates how many isams and isam_iters uses this block. The block can become unloaded only when the counter equals zero. 

			void* ptr_; // Pointer to the beginning of the block.

			Key indexer_; // Costs some memory, but increases performance by a bunch. In fact, this is the key of the first pair. It is saved in the memory. But we want to access this without a need of loading the block.
			
			std::pair<Key, T>* reserve_position(Key key) {

				// Traverse block from the end, move items to the right until the ideal position is not found. Then, return pointer to this position.

				size_t i = ++(*((size_t*)ptr_)) - 1;

				for (; i != 0 && Comparer()(key, data()[i - 1].first); --i) {
					data()[i] = std::move(data()[i - 1]);
					memset(data() + i - 1, 0, sizeof(decltype(data()[i - 1])));
				}

				return &data()[i];
			}

			std::pair<Key, T>* insert(Key key) {

				if (is_full()) {
					return nullptr;
				}

				std::pair<Key, T>* pos = reserve_position(key);

				return new (pos) std::pair<Key, T>(key, T()); // If the block if full, returns nullptr. Otherwise construct a pair at the specific (freed) address.
			}

		public:
			using iterator = std::pair<const Key, T>*;
			
			block_viewer(size_t block_size) : block_size_(block_size), ptr_(nullptr), counter_(0) {  }

			block_viewer(const block_viewer&) = delete;
			block_viewer(block_viewer&&) = delete;

			block_viewer& operator=(const block_viewer&) = delete;
			block_viewer& operator=(block_viewer&&) = delete;

			std::unique_ptr<block_viewer> split(Key key, bool fallback = false) { // Splits the current block into two parts of equal size.

				if (counter_ == 0)
					throw std::runtime_error("Attempt to manipulate with unloaded block.");

				std::unique_ptr<block_viewer> new_block = std::make_unique<block_viewer>(this->capacity());

				size_t records_to_move = size() / 2;

				if (size() == 1 && Comparer()(key, data()[0].first) && fallback) { // Fallback means the first block. The first block must be non-full all the time (when reorganized).
					records_to_move = 1;
				}

				new_block->load();

				std::memcpy(new_block->data(), data() + size() - records_to_move, records_to_move * sizeof(decltype(*data())));
				std::memset(data() + size() - records_to_move,                 0, records_to_move * sizeof(decltype(*data())));

				*((size_t*)ptr_) = size() - records_to_move; // Sets new size for the old block.

				*((size_t*)new_block->ptr_) = records_to_move; // Sets new size for the new block.

				new_block->indexer_ = new_block->size() == 0 ? key : new_block->data()[0].first;

				new_block->unload();

				return std::move(new_block);
			}

			block_viewer* load() { // Loads block into the memory.

				if (counter_++ == 0) { // Block not loaded yet. Load it.
					bool first_loading = ptr_ == nullptr; // If it is a first load, we have to set block capacity. This value should be written directly in the block (not in the viewer) if it was a block located in hard-drive...

					if (first_loading) {
						size_t block_size = block_size_; // Will be rewritten in union.
						block_id_ = block_provider::create_block(2 * sizeof(size_t) + block_size_ * sizeof(std::pair<Key, T>));

						ptr_ = block_provider::load_block(block_id_);

						*((size_t*)ptr_ + 1) = block_size; // Sets capacity.
						*((size_t*)ptr_) = 0; // Sets size.
					}

					else {
						ptr_ = block_provider::load_block(block_id_);
					}
				}

				return this;
			}

			void unload() { // Stores block into the memory.	

				if (counter_ == 0) {
					return;
				}

				else if (--counter_ == 0) { // Store the block only when it is not used by iterator or isam.
					block_provider::store_block(block_id_, ptr_);
				}
			}

			iterator begin() { // Something like random access iterator.		

				return reinterpret_cast<std::pair<const Key, T>*>(data()); // Reinterpret cast to convert from non-const to const members.
			}

			iterator end() { // Something like random access iterator.

				return reinterpret_cast<std::pair<const Key, T>*>(data() + size()); // Reinterpret cast to convert from non-const to const members.
			}

			bool is_loaded() const { // Indicates whether the block is loaded into the memory.

				return counter_ > 0;
			}

			size_t size() const { // Returns current size of the block.

				if (counter_ == 0)
					throw std::runtime_error("Attempt to manipulate with unloaded block.");

				return *((size_t*)ptr_);
			}

			Key get_indexer() const { // Gets indexer for the block.

#if false // Gets indexer from the memory.

				if (counter_ == 0)
					throw std::runtime_error("Attempt to manipulate with unloaded block.");

				return data()[0].first;

#else // Gets indexer from the object.

				return indexer_;
#endif
			}

			std::pair<Key, T>* data() const { // Returns pointer to .data().

				if (counter_ == 0)
					throw std::runtime_error("Attempt to manipulate with unloaded block.");

				return ((std::pair<Key, T>*)((size_t*)ptr_ + 2));
			}

			size_t capacity() const { // Returns total capacity of the block.

				if (counter_ == 0)
					throw std::runtime_error("Attempt to manipulate with unloaded block.");

				return *((size_t*)ptr_ + 1);
			}

			std::pair<iterator, bool> get(Key index) { // Accesses element with given key in the block. Semantics similar to std containers' insert.

				if (counter_ == 0)
					throw std::runtime_error("Attempt to manipulate with unloaded block.");
				
#if STL_BINARY_SEARCH // STL version of binary search.

				// Suggested implementation of std::binary search (with returning found item). Binary search in STL returns true/false only.

				iterator found = std::lower_bound(begin(), end(), std::make_pair(index, T()), [=](const std::pair<const Key, T>& lhs, const std::pair<const Key, T>& rhs) { return Comparer()(lhs.first, rhs.first); });

				if (!(found == end()) && !(Comparer()(index, found->first))) { // Found.

					return std::make_pair(found, true);

				}

				else { // Otherwise insert the new item.

					auto inserted = reinterpret_cast<iterator>(insert(index)); // Reinterpret cast to convert from non-const to const members.

					return std::make_pair(inserted, inserted != nullptr); // If the item was not inserted (block is full), returns false.

				}


#else

				if (size() != 0) { // Else no items in the block. Insert new item with default value in that case

					// We can use binary search O(log n) algorithm to find the key.

					size_t left = 0;
					size_t right = size() - 1;

					// Note that equality operator can be simulated by lower than (<) operator using !((x < y) || (y < x)) <=> x == y.

					while (left <= right) {
						size_t middle = left + (right - left) / 2;

						// Check if key is present at mid.
						if (!(Comparer()(data()[middle].first, index) || Comparer()(index, data()[middle].first))) {
							return &data()[middle];
						}

						// If key greater, ignore left half.
						if (Comparer()(data()[middle].first, index)) {
							if (middle == size_t(-1)) // Overflow.
								break;
							else
								left = middle + 1;
						}

						// If key is smaller, ignore right half.
						else {
							if (middle == 0) // Overflow.
								break;
							else
								right = middle - 1;
						}
					}
				}

				// At this position, key was not found. Insert default.

				return insert(index);

#endif

			}

			bool is_full() const { // Indicates whether the block is full.

				if (counter_ == 0)
					throw std::runtime_error("Attempt to manipulate with unloaded block.");

				return size() == capacity();
			}

			~block_viewer() {

				if (ptr_ != nullptr) {
					if (counter_ > 0) {
						block_provider::store_block(block_id_, ptr_);
					}
					block_provider::free_block(block_id_);
				}
			}

		};

		template <typename Item>
		class load_guard {
			// This is something (but not much) similar to std::lock_guard, in combination with std::unique_(something). I had no idea for better name. 
			// Loads selected block and unloads it when disposing. Guarantees no memory leaks (no block left in memory without unloading it).
			// One load guard for each isam and isam iterator. Load guard garantees the only one block is loaded at a time.
		private:
			Item* item_;
		public:
			load_guard() : item_(nullptr) {  }

			load_guard(Item& item) : item_(item.load()) {  }

			load_guard(const load_guard& other) : item_(const_cast<load_guard&>(other).item_->load()) {  }

			load_guard(load_guard&&) = delete;

			void reset(Item& item) { // Replaces item in the load guard. That means, unload the old one and load the new one.

				if (item_ != nullptr) {
					item_->unload();
				}

				item_ = item.load();
			}

			Item& get() const { // Gets block that is currently loaded in the guard.

				return *item_;
			}

			load_guard& operator= (const load_guard&) = delete;

			load_guard& operator= (load_guard&&) = delete;

			~load_guard() { // While disposing, unload the item.
				if (item_ != nullptr) {
					item_->unload();
				}
			}
		};

		template <typename Key, typename T, typename Comparer = std::less<Key>>
		class index_tree {

			using block = impl::block_viewer<Key, T, Comparer>;
			using index_tree_map = std::map<Key, std::unique_ptr<block>, Comparer>; // Unique pointers were chosen because we need observers to these blocks, and item in std::map changes its position in a time.

		private:
			block default_block_; // Default fallback for interval (-inf, first key in the index tree].
			index_tree_map tree_; // Index tree. Perfect RB tree, although B+ tree might be better. All operations in O(log n). Set may be sufficient as well, but then I think it would not be readable at all. We would only save sizeof(Key) memory per a block.
						
			load_guard<block> loaded_block_;

		public:

			// Basic iterator class for index tree satisfying our needs.
			class iterator {
			private:
				bool first_item_;

				load_guard<block> loaded_block_;

				typename index_tree_map::iterator map_iterator_;
				typename index_tree_map::iterator map_iterator_end_;

				typename block::iterator block_iterator_;

				bool end_reached_; // For isam iterator efficiency.
			public:
				typedef iterator self_type;
				typedef typename std::pair<const Key, T> value_type;
				typedef typename std::pair<const Key, T>& reference;
				typedef typename std::pair<const Key, T>* pointer;
				typedef std::forward_iterator_tag iterator_category;
				typedef ptrdiff_t difference_type;

				iterator() {  }

				iterator(index_tree& tree, bool end) : first_item_(!end), map_iterator_(end ? tree.tree_.end() : tree.tree_.begin()), end_reached_(end), 
				    map_iterator_end_(tree.tree_.end()) {

					if (!end) { // Begin iterator.
						loaded_block_.reset(tree.default_block_);
						block_iterator_ = loaded_block_.get().begin();
					}

					else { // End iterator.
						loaded_block_.reset(tree.tree_.empty() ?
							tree.default_block_ :
							*(tree.tree_.rbegin())->second);
						block_iterator_ = loaded_block_.get().end();
					}
				}

				iterator(const self_type& other) : map_iterator_(other.map_iterator_), first_item_(other.first_item_), block_iterator_(other.block_iterator_), loaded_block_(other.loaded_block_), end_reached_(other.end_reached_), map_iterator_end_(other.map_iterator_end_) {  }

				bool end_reached() const { // For isam iterator efficiency (no need to compare with end iterator).

					return end_reached_;
				}

				self_type& operator=(const self_type& other) {

					if (&other == this)
						return *this;

					map_iterator_ = other.map_iterator_;
					first_item_ = other.first_item_;
					block_iterator_ = other.block_iterator_;
					map_iterator_end_ = other.map_iterator_end_;
					end_reached_ = other.end_reached_;
					loaded_block_.reset(other.loaded_block_.get());

					return *this;
				}

				bool operator== (const self_type& other) const {

					return this->operator->() == other.operator->();
				}

				bool operator!= (const self_type& other) const {

					return !(*this == other);
				}

				reference operator*() const {

					return *(this->operator->());
				}

				pointer operator->() const {

					return block_iterator_;
				}

				self_type& operator++() {

					if (first_item_) { // First (fallback) block.
						if (++block_iterator_ == loaded_block_.get().end()) { // End of the default block reached.
							first_item_ = false;

							if (map_iterator_ != map_iterator_end_) {
								loaded_block_.reset(*map_iterator_->second);
								block_iterator_ = loaded_block_.get().begin();
							}

							else {
								end_reached_ = true;
							}
						}
					}

					else {
						if (++block_iterator_ == loaded_block_.get().end()) { // End of the iterated block reached.

							if (++map_iterator_ != map_iterator_end_) {
								loaded_block_.reset(*map_iterator_->second);
								block_iterator_ = loaded_block_.get().begin();
							}

							else { // End of the map reached. In the next iteration, we are iterating behind the last block.
								end_reached_ = true;
							}

						}
					}

					return *this;
				}

				self_type operator++(int) {

					const auto result(*this);
					++(*this);
					return result;
				}
				
				friend void swap(self_type& lhs, self_type& rhs) {
					std::swap(lhs, rhs);
				}
			};

			index_tree(size_t block_size) : default_block_(block_size), loaded_block_(default_block_) { }

			block& get_block_by_key(Key key) {

				if (tree_.empty())
					return default_block_;

				else {
					auto it = tree_.upper_bound(key); // Find appropriate block. Finds first block which key is GREATER THAN given key.

					if (it == tree_.begin()) { // If it is a beginning of the tree, we have to check whether the key does not fit into the fallback block.
						return key < it->first ? default_block_ : *it->second;
					}

					else { // If not, we still have to check the end. If the end was reached, no key in the map is considered to go after given key. In that case, selects the last block via rbegin().
						return it == tree_.end() ? *tree_.rbegin()->second : *(--it)->second;
					}
				}
			}

			block& split_block_by_key(Key key) { // And returns reference to the new block.

				reload_block(get_block_by_key(key));

				std::unique_ptr<block> splitted_block = std::move(loaded_block_.get().split(key, &loaded_block_.get() == &default_block_));

				block& new_block = *tree_.insert(std::make_pair(splitted_block->get_indexer(), std::move(splitted_block))).first->second;
								
				reload_block(get_block_by_key(key)); // Loads block for current key.

				return new_block;
			}

			std::pair<std::pair<const Key, T>*, bool> find_or_create(Key key) { // Semantics similar to std containers' insert.

				reload_block(get_block_by_key(key));

				auto pos = loaded_block_.get().get(key);

				return pos;
			}

			void reload_block(block& b) {

				if (&loaded_block_.get() != &b) { // If the wanted block is not loaded, load it.
					loaded_block_.reset(b);
				}
			}

			block& get_loaded_block() const {

				return loaded_block_.get();
			}

			std::pair<Key, bool> try_get_key(const block& b) const { // If it is a default block, returns false. 

				return &b == &default_block_ ? std::make_pair(Key(), false) : std::make_pair(b.get_indexer(), true);
			}

			iterator begin() {

				return iterator(*this, false);
			}

			iterator end() {

				return iterator(*this, true);
			}
		};

		template <typename Key, typename T, typename Comparer = std::less<Key>>
		class overflow_space {

			using overflow_space_map = std::map<Key, T, Comparer>;

		private:
			overflow_space_map overflow_space_;

			const size_t capacity_; // Max capacity of the overflow space. When size reaches the capacity, reorganize.
		public:

			using iterator = typename overflow_space_map::iterator;

			overflow_space(size_t capacity) : capacity_(capacity) {  }

			void flush(index_tree<Key, T, Comparer>& target) { // Reorganization takes place here.

				for (auto&& item : overflow_space_) {

					target.reload_block(target.get_block_by_key(item.first));

					if (target.get_loaded_block().is_full()) {
						target.split_block_by_key(item.first);
					}

					target.get_loaded_block().get(item.first).first->second = std::move(item.second);

				}

				overflow_space_.clear();
			}

			bool is_full() const {

				return capacity_ == overflow_space_.size();
			}

			std::pair<std::pair<const Key, T>*, bool> find_or_create(Key key) { // Semantics similar to std containers' insert.

				auto found = overflow_space_.find(key);

				if (found != overflow_space_.end()) {
					return std::make_pair(found.operator->(), true);
				}

				else if (overflow_space_.size() == capacity_) {
					return std::make_pair(nullptr, false);
				}

				else {
					auto res = overflow_space_.insert(std::make_pair(key, T()));
					return std::make_pair(res.first.operator->(), res.second);
				}
			}

			size_t size() const {

				return overflow_space_.size();
			}

			size_t capacity() const {

				return capacity_;
			}

			iterator begin() {

				return overflow_space_.begin();
			}

			iterator end() {

				return overflow_space_.end();
			}
		};
	}

	template <typename Key, typename T, typename Comparer = std::less<Key>>
	class isam {
		static_assert(std::is_fundamental<Key>::value, "The key used in isam must be simple value type.");
		static_assert(std::is_default_constructible<T>::value, "The value must be default constructible.");
	private:
		impl::overflow_space<Key, T, Comparer> overflow_space_;

		impl::index_tree<Key, T, Comparer> index_tree_;

		template <typename KeyValuePair = std::pair<const Key, T>>
		// This way, we can make const iterator easily. 
		// KeyValuePair = std::pair<const Key, T> is normal iterator.  
		// KeyValuePair = const std::pair<const Key, const T> is const iterator.
		class isam_iterator {
			static_assert(std::is_same<std::pair<const Key,       T>, typename std::remove_const<KeyValuePair>::type>::value ||
				          std::is_same<std::pair<const Key, const T>, typename std::remove_const<KeyValuePair>::type>::value,
				          "Template argument for isam and its iterator must be the same.");
		private:
			enum state {
				end,
				block,
				overflow
			};
			
			typename impl::index_tree<Key, T, Comparer>::iterator block_iterator_; // We do not need to save end of this iterator, it is included in it itself (end_reached() method).

			typename impl::overflow_space<Key, T, Comparer>::iterator overflow_space_iterator_;

			state state_; // Current state of the iterator.

			typename impl::overflow_space<Key, T, Comparer>::iterator overflow_space_iterator_end_;


		public:
			typedef isam_iterator self_type;
			typedef KeyValuePair value_type;
			typedef KeyValuePair& reference;
			typedef KeyValuePair* pointer;
			typedef std::forward_iterator_tag iterator_category;
			typedef ptrdiff_t difference_type;

			isam_iterator() {  }

			isam_iterator(const isam& is, bool end) : overflow_space_iterator_(end ? const_cast<isam&>(is).overflow_space_.end() : const_cast<isam&>(is).overflow_space_.begin()),
				overflow_space_iterator_end_(const_cast<isam&>(is).overflow_space_.end()) {

				if (!end) { // Begin iterator.

					block_iterator_ = const_cast<isam&>(is).index_tree_.begin(); // Set block iterator to the beginning.

					if (block_iterator_.end_reached() && overflow_space_iterator_ == overflow_space_iterator_end_) { // No items in blocks. No items in overflow space. Sets end.
						state_ = state::end;
					}
					else if (const_cast<isam&>(is).overflow_space_.size() == 0 || Comparer()(block_iterator_->first, overflow_space_iterator_->first)) { // Nothing in overflow space OR first item in block is lower than the first one in overflow space.
						state_ = state::block;
					}
					else { // First item in overflow space is lower than the first one in block.
						state_ = state::overflow;
					}

				}

				else { // End iterator.
					block_iterator_ = const_cast<isam&>(is).index_tree_.end();
					state_ = state::end;
				}
			}

			isam_iterator(const self_type& other) : state_(other.state_), overflow_space_iterator_(other.overflow_space_iterator_), block_iterator_(other.block_iterator_),
				overflow_space_iterator_end_(other.overflow_space_iterator_end_) {  }

			self_type& operator=(const self_type& other) {

				if (&other == this)
					return *this;
				
				overflow_space_iterator_ = other.overflow_space_iterator_;
				block_iterator_ = other.block_iterator_;
				state_ = other.state_;

				overflow_space_iterator_end_ = other.overflow_space_iterator_end_;

				return *this;
			}

			bool operator== (const self_type& other) const {

				return this->operator->() == other.operator->();
			}

			bool operator!= (const self_type& other) const {

				return !(*this == other);
			}

			reference operator*() const {

				return *(this->operator->());
			}

			pointer operator->() const {

				switch (state_) {
				case state::end:
					[[fallthrough]];
				case state::block:
					return reinterpret_cast<pointer>(block_iterator_.operator->()); // Reinterpret cast to convert from non-const to const members.
				case state::overflow:
					return reinterpret_cast<pointer>(overflow_space_iterator_.operator->()); // Reinterpret cast to convert from non-const to const members.
				default:
					throw std::logic_error("Invalid state.");
				}
			}

			self_type& operator++() {

				// Iterator is represented by a finite automata, changing its state when increasing the iterator.

				switch (state_) {

				case state::end: // Cannot move from this state. Just increase the iterator.
					++block_iterator_;
					break;

				case state::block:

					++block_iterator_;

					if (block_iterator_.end_reached() && overflow_space_iterator_ == overflow_space_iterator_end_) {

						state_ = state::end;

					}

					else if (block_iterator_.end_reached() || (overflow_space_iterator_ != overflow_space_iterator_end_ && Comparer()(overflow_space_iterator_->first, block_iterator_->first))) {

						state_ = state::overflow;

					}

					break;

				case state::overflow:

					++overflow_space_iterator_;

					if (block_iterator_.end_reached() && overflow_space_iterator_ == overflow_space_iterator_end_) {

						state_ = state::end;

					}

					else if (overflow_space_iterator_ == overflow_space_iterator_end_ || (!block_iterator_.end_reached() && Comparer()(block_iterator_->first, overflow_space_iterator_->first))) {

						state_ = state::block;

					}


					break;
				}

				return *this;
			}

			self_type operator++(int) {

				const auto result(*this);
				++(*this);
				return result;
			}

			friend void swap(self_type& lhs, self_type& rhs) {
				std::swap(lhs, rhs);
			}
		};
	public:

		using iterator       = isam_iterator<      std::pair<const Key,       T>>;
		using const_iterator = isam_iterator<const std::pair<const Key, const T>>;

		isam(size_t block_size = 32, size_t overflow_space_size = 8) : index_tree_(block_size), overflow_space_(overflow_space_size) {

			if (block_size == 0) {
				throw std::invalid_argument("Block size must be greater than zero.");
			}
		}

		isam(const isam&) = delete; // I think default copy-constructor would work like a charm, but I cannot guarantee it without testing it.

		isam& operator= (const isam&) = delete;

		T& operator[] (Key key) {

			auto search_in_blocks = index_tree_.find_or_create(key);

			if (search_in_blocks.second) { // Item was found somewhere in the blocks.
				return search_in_blocks.first->second;
			}

			else if (overflow_space_.capacity() == 0) { // Item was not inserted due to full block.
				index_tree_.split_block_by_key(key);
				return index_tree_.find_or_create(key).first->second;
			}
			
			auto search_in_overflow_space = overflow_space_.find_or_create(key);

			if (search_in_overflow_space.second) { // Item was found somewhere in the overflow space.
				return search_in_overflow_space.first->second;
			}

			else if (overflow_space_.is_full()) { // Item not inserted, the overflow space is full. Reorganize and insert again.
				overflow_space_.flush(index_tree_);
				return this->operator[](key);
			}

			throw std::logic_error("Record could not be found."); // Should never be reached (debug case).
		}

		iterator begin() {

#if FLUSH_OVERFLOW_IN_ITERATOR_CTOR  
			overflow_space_.flush(index_tree_);
#endif
			return iterator(*this, false);

		}

		iterator end() {

#if FLUSH_OVERFLOW_IN_ITERATOR_CTOR  
			overflow_space_.flush(index_tree_);
#endif
			return iterator(*this, true);

		}

		const_iterator cbegin() const {

#if FLUSH_OVERFLOW_IN_ITERATOR_CTOR  
			overflow_space_.flush(index_tree_);
#endif
			return const_iterator(*this, false);

		}

		const_iterator cend() const {

#if FLUSH_OVERFLOW_IN_ITERATOR_CTOR  
			overflow_space_.flush(index_tree_);
#endif
			return const_iterator(*this, true);

		}

	};
}

#endif // !isam_hpp