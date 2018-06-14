#pragma once

#include <chrono>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

class Stopwatch
{
public:
	Stopwatch();
	~Stopwatch();
	static std::chrono::high_resolution_clock::time_point now();
	int start();
	int stop();
	double getDuration(int count) const;
	double getFullDuration() const;
	std::string getFullDurationString() const;
	std::string getDurationString(int count) const;
	static std::string getFormatted(double nmbr);
private:
	int m_count;
	std::vector<std::chrono::high_resolution_clock::time_point> m_start;
	std::vector<std::chrono::high_resolution_clock::time_point> m_stop;
};

