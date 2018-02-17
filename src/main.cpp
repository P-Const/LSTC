#include <iostream>
#include "lstc.hpp"

int main(int argc, char **argv) {
	int num_cats = 8;
	int num_closest = 5;
	int num_tfidfs = 4; // default
	omp_set_dynamic(0);
	for (int i = 1; i < argc; ++i) {
		if (strcmp(argv[i], "cats") == 0) {
			num_cats = strtol(argv[i + 1], nullptr, 10);
		}
		else if (strcmp(argv[i], "nghbrs") == 0) {
			num_closest = strtol(argv[i + 1], nullptr, 10);
		}
		else if (strcmp(argv[i], "tfidfs") == 0) {
			num_tfidfs = strtol(argv[i + 1], nullptr, 10);
		}
		else if (strcmp(argv[i], "threads") == 0) {
			omp_set_num_threads(strtol(argv[i + 1], nullptr, 10));
		}
	}
	std::cout << "Input paramters: " << num_cats << ", " << num_closest << ", " << num_tfidfs << std::endl;

	auto test_file = "data/test.csv";
	auto train_file = "data/train.csv";

	//auto test_file = "test.csv";
	//auto train_file = "train.csv";

	auto deja_vu = new int[max_label + 1];
	for (size_t i = 0; i <= max_label; ++i) {
		*(deja_vu + i) = 0;
	}

	std::vector<Doc> test_data;
	std::vector<DocData> train_data;

	test_data.reserve(test_size);
	train_data.reserve(train_size);

	if (!FillTestSet(test_file, test_data)) {
		std::cout << "Error: test file could not be opened!" << std::endl;
		return 1;
	}

	if (!FillTrainSet(train_file, train_data)) {
		std::cout << "Error: train file could not be opened!" << std::endl;
		return 1;
	}

	auto result = KnnTfidf(deja_vu, test_data, train_data, num_cats, num_closest, num_tfidfs);

	WriteOutput(result, "output/output.csv");
	delete[] deja_vu;
	deja_vu = nullptr;
	free(result);
	return 0;
}