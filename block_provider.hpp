#pragma once
#include <unordered_map>
#include <cstring>

namespace block_provider
{
	size_t last_block_id_ = 1, read_count_ = 0, block_in_memory_ = 0;
	std::unordered_map<size_t, void*> blocks_;

	inline size_t create_block(size_t block_size)
	{
		auto block_id = last_block_id_++;
		blocks_[block_id] = malloc(block_size);
		memset(blocks_[block_id], 0, block_size);
		return block_id;
	}

	inline void* load_block(size_t block_id)
	{
		++read_count_;
		++block_in_memory_;
		
		//if not exist
		if (blocks_.find(block_id) == blocks_.end())
		{
			return nullptr;
		}
		return blocks_[block_id];
	}

	inline void store_block(size_t block_id, void* block_ptr)
	{
		--block_in_memory_;
		blocks_[block_id] = block_ptr;
	}

	inline void free_block(size_t block_id)
	{
		free(blocks_[block_id]);
		blocks_.erase(block_id);
	}
}