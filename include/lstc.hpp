#pragma once

#ifdef _WIN32
#define strtok_r strtok_s
#endif // _WIN32

#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <omp.h>

const size_t test_size	= 452167;
const size_t train_size = 2365436;
const size_t max_label	= 2085164;

extern std::vector<char *> test_data, train_data;
extern int *deja_vu;

enum class BufStatus {
	FILLED_OK,
	EMPTY_ERROR,
	RELEASED
};

struct Rel {
	int feat;
	int val;
	explicit Rel(char *&);
};

struct Buf {
	char *content;
	BufStatus status;
	explicit Buf(const char *&);
	~Buf();
};

struct Doc {
	int id;
	int label;
	std::vector<Rel> features;
	Doc();
	explicit Doc(char *&);
};

struct DocData {
	int num;
	std::vector<int> labels;
	std::vector<Rel> features;
	DocData(int, char *&);
};

struct DocRec {
	int id;
	std::vector<int> labels;
	DocRec(int, std::vector<int> &);
};

inline Rel::Rel(char *&token) {
	char *c_val;
	this->feat = strtol(token, &c_val, 10);
	this->val = strtol(c_val + 1, nullptr, 10);
}

inline Doc::Doc(char *&doc_row) {
	char *data_context, *c_label;
	auto token = strtok_r(doc_row, " ", &data_context);
	this->id = strtol(token, &c_label, 10);
	this->label = strtol(c_label + 1, nullptr, 10);
	this->features.clear();
	for (token = strtok_s(nullptr, " ", &data_context); token; token = strtok_r(nullptr, " ", &data_context)) {
		this->features.emplace_back(Rel(token));
	}
}

inline DocData::DocData(int num, char *&data_row) {
	this->num = num;
	char *data_context;
	this->labels.clear();
	this->features.clear();
	auto features_reached = false;
	for (auto token = strtok_r(data_row, " ", &data_context); token; token = strtok_r(data_context, " ", &data_context)) {
		if (features_reached) {
		jmp_reached:
			this->features.emplace_back(Rel(token));
		}
		else if (strchr(token, ':')) {
			features_reached = true;
			goto jmp_reached;
		}
		else {
			this->labels.emplace_back(strtol(token, nullptr, 10));
		}
	}
}

inline DocRec::DocRec(int id, std::vector<int> &vl) : id(id), labels(vl) {
}

bool FillTestSet(const char *, std::vector<Doc> &);
bool FillTrainSet(const char *, std::vector<DocData> &);
bool WriteOutput(const DocRec *, const std::string &);

std::vector<std::pair<int, double>>	MaxTfidfs(const Doc &, const std::vector<DocData> &, int *, int);
std::vector<const DocData *> SuitableSubset(const std::vector<DocData> &, const std::vector<std::pair<int, double>> &);
std::vector<std::pair<double, const DocData*>> MinDistances(const Doc *, const std::vector<const DocData *> &, int);
std::vector<std::pair<int, int>> MaxFrequencies(const std::vector<std::pair<double, const DocData *>> &, int);
DocRec *KnnTfidf(int *, const std::vector<Doc> &, const std::vector<DocData> &, int, int, int);

