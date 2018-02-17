#include "lstc.hpp"

using namespace std;

Buf::Buf(const char *&file_name) {
	ifstream ifs(file_name);
	if (ifs) {
		ifs.seekg(0, ios::end);
		auto length = ifs.tellg();
		ifs.seekg(0, ios::beg);
		this->content = new char[length];
		ifs.read(this->content, length);
		ifs.close();
		this->status = BufStatus::FILLED_OK;
	}
	else {
		this->status = BufStatus::EMPTY_ERROR;
	}
}

Buf::~Buf() {
	delete[] this->content;
	this->content = nullptr;
	this->status = BufStatus::RELEASED;
}

bool FillTestSet(const char *file_name, vector<Doc> &test_data) {
	Buf buf(file_name);
	if (buf.status == BufStatus::FILLED_OK) {
		char *context;
		strtok_r(buf.content, "\r\n", &context);	// header row
		for (auto doc_row = strtok_r(nullptr, "\r\n", &context); doc_row; doc_row = strtok_r(nullptr, "\r\n", &context)) {
			test_data.emplace_back(Doc(doc_row));
		}
		return true;
	}
	else
		return false;
}

bool FillTrainSet(const char *file_name, vector<DocData> &train_data) {
	Buf buf(file_name);
	if (buf.status == BufStatus::FILLED_OK) {
		int num = 0;
		char *context;
		strtok_r(buf.content, "\r\n", &context);	// header row
		for (auto data_row = strtok_r(nullptr, "\r\n", &context); data_row; data_row = strtok_r(nullptr, "\r\n", &context)) {
			train_data.emplace_back(DocData(++num, data_row));
		}
		return true;
	}
	else {
		return false;
	}
}

vector<pair<int, double>> MaxTfidfs(const Doc &instance, const vector<DocData> &train_data, int *deja_vu, int max_tfidfs_count) {
	vector<pair<int, double>> res(max_tfidfs_count);
	auto sum_n = accumulate(instance.features.begin(), instance.features.end(), 0, [](int sum, const Rel &rel) { 
		return sum + rel.val; 
	});
	auto q = 0U;
	for (auto &rel : instance.features) {
		auto tf = static_cast<double>(rel.val) / sum_n;
		if (deja_vu[rel.feat - 1] == 0) {
			q = count_if(train_data.begin(), train_data.end(), [&rel](const DocData &r) {
				return find_if(r.features.begin(), r.features.end(), [&rel](const Rel &p) {
					return p.feat == rel.feat;
				}) != r.features.end();
			});
			if (q != 0) {
#pragma omp critical
				{
					deja_vu[rel.feat - 1] = q;
				}
			}
		}
		else {
			q = deja_vu[rel.feat - 1];
		}
		auto idf = 0.;
		if (q != 0) {
			idf = log(static_cast<double>(train_size) / q);
			auto tfidf = tf * idf;
			auto minit = min_element(res.begin(), res.end(), [](const pair<int, double> &p1, const pair<int, double> &p2) {
				return p1.second < p2.second; 
			}); 
			if (minit->second < tfidf) {
				*minit = make_pair(rel.feat, tfidf);
			}
			q = 0;
		}
	}
	sort(res.begin(), res.end(), [](const pair<int, double> &p1, const pair<int, double> &p2) {
		return p1.second > p2.second; 
	});
	return res;
}

vector<const DocData*> SuitableSubset(const vector<DocData> &train_data, const vector<pair<int, double>> &max_tfidfs) {
	vector<const DocData*> v(0);
	for (auto &data : train_data) {
		for (auto &rel : data.features) {
			if (find_if(max_tfidfs.begin(), max_tfidfs.end(), [&rel](const pair<int, double> &p) {
				return p.first == rel.feat;
			}) != max_tfidfs.end()) {
				v.push_back(&data);
				break;
			}
		}
	}
	return v;
}

vector<pair<double, const DocData*>> MinDistances(const Doc *instance, const vector<const DocData *> &entrants, int closest_count) {
	vector<pair<double, const DocData *>> v(0);
	for (auto &s_i : entrants) {
		auto size = 0U;
		for (auto &term : instance->features) {
			size += count_if(s_i->features.begin(), s_i->features.end(), [&term](const Rel &p) {
				return term.feat == p.feat;
			});
		}
		auto distance =
			static_cast<double>(instance->features.size() + s_i->features.size() - 2 * size) /
			(instance->features.size() + s_i->features.size() - size);
		v.push_back({ distance, s_i });
	}
	sort(v.begin(), v.end(), [](const pair<double, const DocData*> &p1, const pair<double, const DocData*> &p2) {
		return p1.first < p2.first;
	});
	if (entrants.size() < closest_count) {
		v.resize(entrants.size());
	}
	else {
		v.resize(closest_count);
	}
	return v;
}

vector<pair<int, int>> MaxFrequencies(const vector<pair<double, const DocData*>> &entrants, int cats_count) {
	vector<pair<int, int>> v(cats_count);
	vector<int> cats;
	for (auto &p : entrants) {
		copy(p.second->labels.begin(), p.second->labels.end(), back_inserter(cats));
	}
	for (auto c : cats) {
		auto q = count(cats.begin(), cats.end(), c);
		if (find_if(v.begin(), v.end(), [c](pair<int, int> &p) {
			return p.first == c;
		}) == v.end()) {
			auto min_cat = min_element(v.begin(), v.end(), [](const pair<int, int> &p1, const pair<int, int> &p2) {
				return p1.second < p2.second;
			});
			if (min_cat->second < q) {
				*min_cat = make_pair(c, q);
			}
		}
	}
	sort(v.begin(), v.end(), [](const std::pair<int, int> &p1, const std::pair<int, int> &p2) {
		return p1.second > p2.second;
	});
	if (cats.size() < cats_count) {
		v.resize(cats.size());
	}
	return v;
}

DocRec* KnnTfidf(int *deja_vu, const vector<Doc> &test_data, const vector<DocData> &train_data,
	int cats_count, int closest_count, int max_tfidfs_count) {
	DocRec *v = reinterpret_cast<DocRec *>(malloc(sizeof(DocRec) * test_size));
	vector<int> labels;
#pragma omp parallel for private(labels)
	for (size_t k = 0; k < test_data.size(); ++k) {
		auto &instance = test_data[k];
		auto temp_tfidfs = MaxTfidfs(instance, train_data, deja_vu, max_tfidfs_count);
		auto temp_suits = SuitableSubset(train_data, temp_tfidfs);
		auto temp_min_dists = MinDistances(&instance, temp_suits, closest_count);
		auto temp_max_freqs = MaxFrequencies(temp_min_dists, cats_count);
		for (auto &r : temp_max_freqs) {
			labels.push_back(r.first);
		}
		*(v + k) = DocRec(instance.id, labels);
	}
	return v;
}

bool WriteOutput(const DocRec *prediction, const string &output_file) {
	ofstream ofs(output_file, ios::out);
	if (ofs.is_open()) {
		ofs << "Id,Predicted" << endl;
		for (unsigned int i = 0; i < test_size; ++i) {
			ofs << (prediction + i)->id << ",";
			if (!(prediction + i)->labels.empty()) {
				for (auto &c : (prediction + i)->labels) {
					if (c != 0)
						ofs << c << " ";
				}
				ofs << endl;
			}
			else {
				ofs << (prediction + i)->id << ",0" << endl;
			}
		}
		ofs.close();
		return true;
	}
	else {
		return false;
	}
}