#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "../diagnostics.h"
#include "finals.h"

finals_stats::finals_stats(target_t target, state_t internals_mask) : target_(target), internals_mask_(internals_mask)
{}

void finals_stats::process_batch(thrust::device_ptr<state_t>, thrust::device_ptr<float>, thrust::device_ptr<float>,
								 thrust::device_ptr<state_t> last_states,
								 thrust::device_ptr<trajectory_status> traj_statuses, int n_trajectories)
{
	process_batch_internal(last_states, traj_statuses, n_trajectories);
}

void finals_stats::process_batch_internal(thrust::device_ptr<state_t> last_states,
										  thrust::device_ptr<trajectory_status> traj_statuses, int n_trajectories)
{
	timer t;
	float copy_sort_reduce_time = 0.f, update_time = 0.f;

	t.start();

	auto fp_pred = [target = target_] __device__(trajectory_status t) {
		if (target == target_t::FIXED)
			return t == trajectory_status::FIXED_POINT;
		else if (target == target_t::FINAL)
			return t == trajectory_status::FINISHED || t == trajectory_status::FIXED_POINT;
		else
			return false;
	};

	size_t finished_trajs_size = thrust::count_if(traj_statuses, traj_statuses + n_trajectories, fp_pred);

	if (finished_trajs_size == 0)
		return;

	thrust::device_vector<state_t> final_states(finished_trajs_size);

	if (target_ == target_t::FINAL)
	{
		auto m = internals_mask_;
		auto states_it =
			thrust::make_transform_iterator(last_states, [m] __host__ __device__(state_t s) { return s & ~m; });

		thrust::copy_if(states_it, states_it + n_trajectories, traj_statuses, final_states.begin(), fp_pred);
	}
	else if (target_ == target_t::FIXED)
	{
		thrust::copy_if(last_states, last_states + n_trajectories, traj_statuses, final_states.begin(), fp_pred);
	}

	thrust::sort(final_states.begin(), final_states.end());

	size_t unique_fixed_points_size = thrust::unique_count(final_states.begin(), final_states.end());

	thrust::device_vector<state_t> unique_fixed_points(unique_fixed_points_size);
	thrust::device_vector<int> unique_fixed_points_count(unique_fixed_points_size);

	thrust::reduce_by_key(final_states.begin(), final_states.end(), thrust::make_constant_iterator(1),
						  unique_fixed_points.begin(), unique_fixed_points_count.begin());

	t.stop();
	copy_sort_reduce_time = t.millisecs();
	t.start();

	std::vector<state_t> h_unique_fixed_points(unique_fixed_points_size);
	std::vector<int> h_unique_fixed_points_count(unique_fixed_points_size);

	thrust::copy(unique_fixed_points.begin(), unique_fixed_points.end(), h_unique_fixed_points.begin());
	thrust::copy(unique_fixed_points_count.begin(), unique_fixed_points_count.end(),
				 h_unique_fixed_points_count.begin());

	for (size_t i = 0; i < unique_fixed_points_size; i++)
	{
		auto it = result_.find(h_unique_fixed_points[i]);

		if (it != result_.end())
			result_[h_unique_fixed_points[i]] += h_unique_fixed_points_count[i];
		else
			result_[h_unique_fixed_points[i]] = h_unique_fixed_points_count[i];
	}

	t.stop();

	update_time = t.millisecs();

	if (print_diags)
	{
		std::cout << "fixed_points> copy_sort_reduce_time: " << copy_sort_reduce_time << "ms" << std::endl;
		std::cout << "fixed_points> update_time: " << update_time << "ms" << std::endl;
	}
}

void finals_stats::visualize(int n_trajectories, const std::vector<std::string>& nodes)
{
	if (target_ == target_t::FINAL)
		std::cout << "final points:" << std::endl;
	else
		std::cout << "fixed points:" << std::endl;

	for (const auto& p : result_)
	{
		std::cout << (float)p.second / (float)n_trajectories << " " << to_string(p.first, nodes) << std::endl;
	}
}
