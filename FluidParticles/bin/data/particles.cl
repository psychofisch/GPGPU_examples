__kernel void cell(
	__global char *elements,
	__global char *tmp,
	const int size_x,
	const int size_y
)
{
	int tidX = get_global_id(0);
	int tidY = get_global_id(1);
	int alive;
	char neighbour;

	if (tidX < size_x && tidY < size_y)
	{
		int neighbours[] = {
			-1,-1,
			-1,0,
			-1,1,
			0,-1,
			0,0,
			0,1,
			1,-1,
			1,0,
			1,1
		};

		int pos = (tidY * size_x) + tidX;
		alive = 0;

		for (int i = 0; i <= 8; ++i)
		{
			if (i == 4)
				continue;

			int newX, newY;
			newX = tidX + neighbours[2 * i];
			newY = tidY + neighbours[2 * i + 1];

			if (newX < 0)
				newX += size_x;
			else if (newX >= size_x)
				newX = 0;

			if (newY < 0)
				newY += size_y;
			else if (newY >= size_y)
				newY = 0;

			neighbour = elements[(newY * size_x) + newX];

			if (neighbour == 'x')
				++alive;

			//optimization
			//if (i == 6 && alive == 0 && elements[pos] == '.')
			//	break;

			//if (alive >= 4)
			//	break;
			//***
		}

		if (elements[pos] == '.' && alive == 3)
			tmp[pos] = 'x';
		else if (elements[pos] == 'x' && (alive >= 4 || alive <= 1))
			tmp[pos] = '.';
		else
			tmp[pos] = elements[pos];
	}
}
