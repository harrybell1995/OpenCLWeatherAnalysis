#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <fstream>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"

typedef int mytype;
std::vector<mytype> A; //create a vector a

vector<mytype> readFile(std::ifstream &ifs) {
	std::string line;
	while (std::getline(ifs, line))
	{
		// Get line from input string stream
		std::istringstream iss(line);
		std::size_t found = line.find_last_of(" ");
		float val = std::stoi(line.substr(found + 1));
		// cout << std::setprecision(2) << std::fixed << val << " ";
		A.push_back(val);
	}
	return A;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	//detect any potential exceptions
	try {

		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);


		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "my_kernels_3.cl");

		cl::Program program(context, sources);

		FILE *data_file = fopen("C:/Users/Harry/Desktop/Tutorial 3/temp_lincolnshire_short.txt", "r");

		while (!feof(data_file)) {
			/*Get the data from the line, split so it can be placed into the correct vector*/

			//create the temporary temperature holder
			float temp;

			//Scanf saves about a minute of execution time over ifstream
			fscanf(data_file, "%*s %*s %*s %*s %*s %f", &temp); //Ignore everything but the last value as we're only interested in the temperature (save memory)

			//Add the temperature to the temperatures vector
			A.push_back(temp);
		}
		fclose(data_file);



		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		cl::Event sum_event;
		cl::Event min_event;
		cl::Event max_event;
		cl::Event deviation_event;
		cl::Event deviation_event2;

		//Part 3 - memory allocation
		//host - input

		//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
		//if the total input length is divisible by the workgroup size
		//this makes the code more efficient
		size_t local_size = 256;

		size_t padding_size = A.size() % local_size;

		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<int> A_ext(local_size-padding_size, 0);
			//append that extra vector to our input
			A.insert(A.end(), A_ext.begin(), A_ext.end());
		}

		size_t input_elements = A.size();//number of input elements
		size_t input_size = A.size()*sizeof(mytype);//size in bytes
		size_t nr_groups = input_elements / local_size;

		//host - output
		std::vector<mytype> B(input_elements);
		std::vector<mytype> C(input_elements);
		std::vector<unsigned int> D(1);
		size_t output_size = B.size()*sizeof(mytype);//size in bytes

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, input_size);				// ^^ ^^ ^^
		cl::Buffer buffer_D(context, CL_MEM_READ_WRITE, 1 * sizeof(unsigned int));	// ^^ ^^ ^^

		//Part 4 - device operations
		
		//4.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory
		queue.enqueueFillBuffer(buffer_C, 0, 0, input_size);				// ^^ ^^ ^^
		queue.enqueueFillBuffer(buffer_D, 0, 0, 1 * sizeof(unsigned int));	// ^^ ^^ ^^

		//4.2 Setup and execute all kernels (i.e. device code)
		cl::Kernel reduce_sum = cl::Kernel(program, "reduce_sum");
		reduce_sum.setArg(0, buffer_A);
		reduce_sum.setArg(1, buffer_B);
		reduce_sum.setArg(2, cl::Local(local_size * sizeof(float)));

		cl::Kernel reduce_min = cl::Kernel(program, "reduce_min");
		reduce_min.setArg(0, buffer_A);
		reduce_min.setArg(1, buffer_B);
		reduce_min.setArg(2, cl::Local(local_size * sizeof(float)));

		cl::Kernel reduce_max = cl::Kernel(program, "reduce_max");
		reduce_max.setArg(0, buffer_A);
		reduce_max.setArg(1, buffer_B);
		reduce_max.setArg(2, cl::Local(local_size * sizeof(float)));
		

//		kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size

		//call all kernels in a sequence
		queue.enqueueNDRangeKernel(reduce_sum, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &sum_event);
		queue.enqueueNDRangeKernel(reduce_min, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &min_event);
		queue.enqueueNDRangeKernel(reduce_max, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &max_event);

		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

		int size = A.size();
		int mean = B[0] / A.size();

		cl::Kernel std_var = cl::Kernel(program, "std_var");
		std_var.setArg(0, buffer_A);
		std_var.setArg(1, buffer_C);
		std_var.setArg(2, mean);
		std_var.setArg(3, size);
		std_var.setArg(4, cl::Local(local_size * sizeof(float)));
		queue.enqueueNDRangeKernel(std_var, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &deviation_event); // Call kernel in sequence
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, input_size, &C[0]); // Recall data from buffer_C to C

		cl::Kernel std_sum = cl::Kernel(program, "std_sum");	// Define kernel, call kernel from .cl file
		std_sum.setArg(0, buffer_C);	// Pass arguement/buffer to kernel
		std_sum.setArg(1, buffer_D);	// ^^ ^^ ^^
		std_sum.setArg(2, cl::Local(local_size * sizeof(mytype)));	// Local memory instantiated to local_size(bytes)
		queue.enqueueNDRangeKernel(std_sum, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &deviation_event2);  // Call kernel in sequence
		queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, 1 * sizeof(unsigned int), &D[0]); // Recall data from buffer_D to D

		float std_mean = (float)D[0] / C.size();
		float std_sqrt = sqrt(std_mean);

		//std::cout << "A = " << A << std::endl;
		std::cout << "Sum = " << B[0] << std::endl;
		std::cout << "Min = " << B[1] << std::endl;
		std::cout << "Max = " << B[2] << std::endl;
		std::cout << "Mean = " << mean << std::endl;
		std::cout << "Deviation = " << std_sqrt << std::endl;
		//std::cout << "hehe = " << mean << std::endl;
		//std::cout << "C = " << B << std::endl;
		//std::cout << "D = " << D[3] << std::endl;
		std::cout << "Sum execution time [ns]:" << sum_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - sum_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Min execution time [ns]:" << min_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - min_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Max execution time [ns]:" << max_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - max_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Deviation variation execution time [ns]:" << deviation_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviation_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Deviation summing execution time [ns]:" << deviation_event2.getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviation_event2.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}
