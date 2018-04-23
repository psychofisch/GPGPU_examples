#include "Stopwatch.h"

Stopwatch::Stopwatch()
	:m_count(0)
{
	
}


Stopwatch::~Stopwatch()
{
}

std::chrono::high_resolution_clock::time_point Stopwatch::now()
{
	return std::chrono::high_resolution_clock::now();
}

int Stopwatch::start()
{
	m_start.push_back(std::chrono::high_resolution_clock::now());
	return m_count;
}

int Stopwatch::stop()
{
	m_stop.push_back(std::chrono::high_resolution_clock::now());
	m_count++;
	return m_count - 1;
}

double Stopwatch::getDuration(int count) const
{
	if (count >= m_stop.size() || count >= m_start.size())
	{
		return 66.6;
	}
	std::chrono::duration<double> d = std::chrono::duration_cast<std::chrono::duration<double>>(m_stop[count] - m_start[count]);
	return d.count();
}

double Stopwatch::getFullDuration()
{
	std::chrono::duration<double> d = std::chrono::duration_cast<std::chrono::duration<double>>(m_stop[m_stop.size()-1] - m_start[0]);
	return d.count();
}

std::string Stopwatch::getFullDurationString()
{
	return getFormatted(getFullDuration());
}

std::string Stopwatch::getDurationString(int count)
{
	double d = getDuration(count);
	return getFormatted(d);
}

std::string Stopwatch::getFormatted(double nmbr)
{
	std::stringstream test;	
	std::chrono::duration<double> diff(nmbr);

	auto hours = std::chrono::duration_cast<std::chrono::hours>(diff);
	diff -= hours;

	auto mins = std::chrono::duration_cast<std::chrono::minutes>(diff);
	diff -= mins;

	auto secs = std::chrono::duration_cast<std::chrono::seconds>(diff);
	diff -= secs;

	auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(diff);

	test << std::setfill('0');
	test << std::setw(2) << hours.count() << ':'
		<< std::setw(2) << mins.count() << ':'
		<< std::setw(2) << secs.count() << '.'
		<< std::setw(3) << millis.count();

	return test.str();
}
