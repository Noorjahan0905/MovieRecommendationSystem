#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <limits>

using namespace std;

struct Movie {
    string name;
    int id;
};

class Recommender {
private:
    vector<Movie> movies;
    vector<vector<int>> ratings;
    map<int, map<int, double>> similarities;
    vector<double> avgRatings;

    void calcAvgRatings() {
        avgRatings.resize(ratings.size(), 0.0);
        for (size_t u = 0; u < ratings.size(); ++u) {
            double sum = 0.0;
            int count = 0;
            for (int r : ratings[u]) {
                if (r > 0) {
                    sum += r;
                    ++count;
                }
            }
            avgRatings[u] = count > 0 ? sum / count : 0.0;
        }
    }

public:
    void load(const string &file) {
        ifstream f(file);
        if (!f.is_open()) {
            cout << "Error opening file " << file << endl;
            return;
        }

        string line;
        getline(f, line);
        stringstream ss(line);
        string m;
        int id = 1;
        while (getline(ss, m, ',')) {
            movies.push_back({m, id++});
        }

        ratings.clear();
        while (getline(f, line)) {
            stringstream ss(line);
            vector<int> userR;
            string r;
            while (getline(ss, r, ',')) {
                userR.push_back(stoi(r));
            }
            ratings.push_back(userR);
        }
        calcAvgRatings();
    }

    double calcPearson(int u1, int u2) {
        double sum1 = 0, sum2 = 0, sum1Sq = 0, sum2Sq = 0, pSum = 0;
        int count = 0;

        for (size_t i = 0; i < ratings[0].size(); ++i) {
            if (ratings[u1][i] && ratings[u2][i]) {
                double r1 = ratings[u1][i] - avgRatings[u1];
                double r2 = ratings[u2][i] - avgRatings[u2];
                sum1 += r1;
                sum2 += r2;
                sum1Sq += r1 * r1;
                sum2Sq += r2 * r2;
                pSum += r1 * r2;
                ++count;
            }
        }

        if (count == 0) return 0;
        double denom = sqrt(sum1Sq * sum2Sq);
        return denom == 0 ? 0 : pSum / denom;
    }

    void calcSimilarities() {
        for (size_t u1 = 0; u1 < ratings.size(); ++u1) {
            for (size_t u2 = u1 + 1; u2 < ratings.size(); ++u2) {
                double sim = calcPearson(u1, u2);
                similarities[u1][u2] = sim;
                similarities[u2][u1] = sim;
            }
        }
    }

    double predict(int user, int movieId) {
        double num = 0, denom = 0;
        bool hasSims = false;

        for (size_t u = 0; u < ratings.size(); ++u) {
            if (u != user && ratings[u][movieId - 1]) {
                auto it = similarities[user].find(u);
                if (it != similarities[user].end() && it->second > 0) {
                    double sim = it->second;
                    num += sim * ratings[u][movieId - 1];
                    denom += abs(sim);
                    hasSims = true;
                }
            }
        }

        if (!hasSims) {
            double movieAvg = 0;
            int count = 0;
            for (size_t u = 0; u < ratings.size(); ++u) {
                if (ratings[u][movieId - 1]) {
                    movieAvg += ratings[u][movieId - 1];
                    ++count;
                }
            }
            return count > 0 ? movieAvg / count : 3;
        }

        return denom == 0 ? 3 : num / denom;
    }

    vector<pair<int, double>> recommend(int user, int n) {
        vector<pair<int, double>> preds;
        for (int i = 0; i < movies.size(); ++i) {
            if (ratings[user][i] == 0) {
                preds.push_back({i + 1, predict(user, i + 1)});
            }
        }

        partial_sort(preds.begin(), preds.begin() + min(n, (int)preds.size()), preds.end(), [](auto &a, auto &b) {
            return a.second > b.second;
        });

        preds.resize(min(n, (int)preds.size()));
        return preds;
    }

    const vector<vector<int>> &getRatings() const {
        return ratings;
    }

    void printRecs(int user, int n) {
        auto recs = recommend(user, n);
        cout << "\nTop " << recs.size() << " recommendations for User " << user + 1 << ":\n";
        for (auto &rec : recs) { // Updated for pre-C++17 compatibility
            cout << movies[rec.first - 1].name << " (Predicted: " << rec.second << ")\n";
        }
        if (recs.size() < n) {
            cout << "Note: Only " << recs.size() << " recommendations are available.\n";
        }
    }

};

int main() {
    Recommender sys;
    sys.load("E:/Internship/movie_rating.csv");
    cout << "Data loaded.\n";

    int choice;
    do {
        cout << "\nMenu:\n1. Recommend Movies\n2. Exit\nChoice: ";
        cin >> choice;

        if (choice == 1) {
            int user, n;
            cout << "User ID (0 to " << sys.getRatings().size() - 1 << "): ";
            cin >> user;
            cout << "Number of recommendations: ";
            cin >> n;
            sys.printRecs(user, n);
        } else if (choice != 2) {
            cout << "Invalid choice.\n";
        }
    } while (choice != 2);

    cout << "Goodbye!\n";
    return 0;
}
