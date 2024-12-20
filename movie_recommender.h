
#ifndef MOVIE_RECOMMENDER_H
#define MOVIE_RECOMMENDER_H

#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <limits>
#include <iomanip>

class MovieRecommender {
public:
    // Constructor
    MovieRecommender(const std::string& filename);

    // Load ratings from CSV file
    bool loadRatingsFromCSV(const std::string& filename);

    // Calculate user similarity using Pearson correlation
    double calculateUserSimilarity(const std::vector<double>& user1,
                                   const std::vector<double>& user2);

    // Predict rating for a specific user and movie
    double predictRating(int userId, int movieId);

    // Get top N movie recommendations for a user
    std::vector<std::pair<int, double>> getTopNRecommendations(int userId, int N);

    // Calculate Root Mean Square Error (RMSE)
    double calculateRMSE();

    // Print ratings matrix for debugging
    void printRatingsMatrix();

private:
    // 2D vector to store ratings matrix
    std::vector<std::vector<double>> ratingsMatrix;

    // Mapping of movie IDs to column indices
    std::unordered_map<int, int> movieIdToIndex;

    // Mapping of user IDs to row indices
    std::unordered_map<int, int> userIdToIndex;
};

#endif // MOVIE_RECOMMENDER_H
