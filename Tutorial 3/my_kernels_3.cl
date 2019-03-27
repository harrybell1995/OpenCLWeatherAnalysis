kernel void reduce_sum(global const int* A, global int* B, local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) {
		atomic_add(&B[0], scratch[lid]);
	}
}

kernel void reduce_min(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) {
			if (scratch[lid] > scratch[lid + i])
			scratch[lid] = scratch[lid + i];}

		barrier(CLK_LOCAL_MEM_FENCE);	// Sync threads
	}

	//copy the cache to output array
	atomic_min(&B[1], scratch[lid]);
}


kernel void reduce_max(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);	// Sync threads

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) {
			if (scratch[lid] < scratch[lid + i])
			scratch[lid] = scratch[lid + i];}

		barrier(CLK_LOCAL_MEM_FENCE);	// Sync threads
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	
	atomic_max(&B[2],scratch[lid]);
	
}

inline int square_int(int a) {
    return a * a;
}

__kernel void std_var(__global const int* A, __global int* C, int mean, int dataSize, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);

	if (id < dataSize)	// Only operate to original data size (before padding)
		scratch[lid] = (A[id] - mean);	// Subtract mean from all values
	
	barrier(CLK_LOCAL_MEM_FENCE);	// Sync threads

	C[id] = (scratch[lid] * scratch[lid]);	// Square the result and output to C vector
}

__kernel void std_sum(__global const int* C, __global int* D, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = C[id];	// Copy global values to local memory

	barrier(CLK_LOCAL_MEM_FENCE);	// Sync threads

	for (int i = 1; i < N; i *= 2) {	// Add all values in kernel
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];	// Incremental Add

		barrier(CLK_LOCAL_MEM_FENCE);	// Sync threads
	}
	scratch[lid] = scratch[lid];	// / 100.f to ensure data is converted back to float

	if (!lid) {
		atomic_add(&D[0],scratch[lid]);	// Deter race conditions (Atomic functions limit variable accessiblity)
	}
}
