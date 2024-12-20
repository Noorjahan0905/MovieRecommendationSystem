#include "movie_recommender.h"

MovieRecommender::MovieRecommender(const std::string& filename) {
    loadRatingsFromCSV(filename);
}

double MovieRecommender::calculateUserSimilarity(const std::vector<double>& user1,
                                                 const std::vector<double>& user2) {
    std::vector<double> commonRatings1, commonRatings2;
    for (size_t i = 0; i < user1.size(); ++i) {
        if (user1[i] > 0 && user2[i] > 0) {
            commonRatings1.push_back(user1[i]);
            commonRatings2.push_back(user2[i]);
        }
    }

    if (commonRatings1.empty()) return 0.0;

    double mean1 = std::accumulate(commonRatings1.begin(), commonRatings1.end(), 0.0) / commonRatings1.size();
    double mean2 = std::accumulate(commonRatings2.begin(), commonRatings2.end(), 0.0) / commonRatings2.size();

    double numerator = 0.0, denominator1 = 0.0, denominator2 = 0.0;
    for (size_t i = 0; i < commonRatings1.size(); ++i) {
        double diff1 = commonRatings1[i] - mean1;
        double diff2 = commonRatings2[i] - mean2;
        numerator += diff1 * diff2;
        denominator1 += diff1 * diff1;
        denominator2 += diff2 * diff2;
    }

    if (denominator1 == 0 || denominator2 == 0) return 0.0;

    return numerator / std::sqrt(denominator1 * denominator2);
}

bool MovieRecommender::loadRatingsFromCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    std::string line;
    std::vector<int> movieIds;
    size_t rowIndex = 0;

    if (std::getline(file, line)) {
        std::istringstream headerStream(line);
        std::string movieIdStr;
        size_t movieIndex = 0;

        std::getline(headerStream, movieIdStr, ',');

        while (std::getline(headerStream, movieIdStr, ',')) {
            int movieId = std::stoi(movieIdStr);
            movieIdToIndex[movieId] = movieIndex++;
            movieIds.push_back(movieId);
        }
    }

    while (std::getline(file, line)) {
        std::istringstream lineStream(line);
        std::string valueStr;
        std::vector<double> userRatings;

        std::getline(lineStream, valueStr, ',');
        int userId = std::stoi(valueStr);
        userIdToIndex[userId] = rowIndex;

        while (std::getline(lineStream, valueStr, ',')) {
            double rating = std::stod(valueStr);
            userRatings.push_back(rating);
        }

        ratingsMatrix.push_back(userRatings);
        rowIndex++;
    }

    file.close();
    return true;
}

double MovieRecommender::predictRating(int userId, int movieId) {
    size_t userIndex = userIdToIndex[userId];
    size_t movieIndex = movieIdToIndex[movieId];

    if (ratingsMatrix[userIndex][movieIndex] > 0) {
        return ratingsMatrix[userIndex][movieIndex];
    }

    double weightedRatingSum = 0.0;
    double similaritySum = 0.0;

    for (size_t i = 0; i < ratingsMatrix.size(); ++i) {
        if (i == userIndex) continue;

        if (ratingsMatrix[i][movieIndex] > 0) {
            double similarity = calculateUserSimilarity(ratingsMatrix[userIndex], ratingsMatrix[i]);
            weightedRatingSum += similarity * ratingsMatrix[i][movieIndex];
            similaritySum += std::abs(similarity);
        }
    }

    return (similaritySum > 0) ? weightedRatingSum / similaritySum : 0.0;
}

std::vector<std::pair<int, double>> MovieRecommender::getTopNRecommendations(int userId, size_t N) {
    size_t userIndex = userIdToIndex[userId];
    std::vector<std::pair<int, double>> movieRatings;

    for (const auto& moviePair : movieIdToIndex) {
        int movieId = moviePair.first;
        size_t movieIndex = moviePair.second;

        if (ratingsMatrix[userIndex][movieIndex] > 0) continue;

        double predictedRating = predictRating(userId, movieId);
        movieRatings.push_back({movieId, predictedRating});
    }

    std::sort(movieRatings.begin(), movieRatings.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    return std::vector<std::pair<int, double>>(
        movieRatings.begin(),
        movieRatings.begin() + std::min(N, movieRatings.size())
    );
}

double MovieRecommender::calculateRMSE() {
    double sumSquaredError = 0.0;
    size_t totalRatings = 0;

    for (size_t userIndex = 0; userIndex < ratingsMatrix.size(); ++userIndex) {
        for (size_t movieIndex = 0; movieIndex < ratingsMatrix[userIndex].size(); ++movieIndex) {
            if (ratingsMatrix[userIndex][movieIndex] == 0) continue;

            int userId = -1, movieId = -1;
            for (const auto& pair : userIdToIndex) {
                if (static_cast<size_t>(pair.second) == userIndex) {
                    userId = pair.first;
                    break;
                }
            }
            for (const auto& pair : movieIdToIndex) {
                if (static_cast<size_t>(pair.second) == movieIndex) {
                    movieId = pair.first;
                    break;
                }
            }

            double predictedRating = predictRating(userId, movieId);
            double actualRating = ratingsMatrix[userIndex][movieIndex];

            sumSquaredError += std::pow(actualRating - predictedRating, 2);
            totalRatings++;
        }
    }

    return std::sqrt(sumSquaredError / totalRatings);
}

void MovieRecommender::printRatingsMatrix() {
    std::cout << "Ratings Matrix:" << std::endl;
    std::cout << "User ID\t";
    for (const auto& moviePair : movieIdToIndex) {
        std::cout << "Movie" << moviePair.first << "\t";
    }
    std::cout << std::endl;

    for (size_t userIndex = 0; userIndex < ratingsMatrix.size(); ++userIndex) {
        for (const auto& pair : userIdToIndex) {
            if (static_cast<size_t>(pair.second) == userIndex) {
                std::cout << pair.first << "\t";
                break;
            }
        }

        for (double rating : ratingsMatrix[userIndex]) {
            std::cout << std::fixed << std::setprecision(1) << rating << "\t";
        }
        std::cout << std::endl;
    }
}
