#pragma once

#include "state.h"
#include "transition_rates.h.generated"

using state_t = state_t_template<states_count>;

enum class trajectory_status : uint8_t
{
	CONTINUE,
	FINISHED,
	FIXED_POINT
};
