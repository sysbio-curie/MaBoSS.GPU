#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LENGTH 80

int main(int argc, char** argv)
{
	if (argc != 3)
	{
		return 1;
	}

	FILE* fin = fopen(argv[1], "r");

	if (ferror(fin))
	{
		fprintf(stderr, "Error opening input file");
		return 1;
	}

	FILE* fout = fopen(argv[2], "w");

	if (ferror(fout))
	{
		fprintf(stderr, "Error opening output file");
		return 1;
	}

	char init_line[] = { "{ " };
	const int offset_length = strlen(init_line);

	char offset_spc[offset_length];

	unsigned char buff[1024];
	char curr_out[64];

	int count, i;
	int line_length = 0;

	memset((void*)offset_spc, (char)32, sizeof(char) * offset_length - 1);
	offset_spc[offset_length - 1] = '\0';

	fprintf(fout, "%s", init_line);

	while (!feof(fin))
	{
		count = fread(buff, sizeof(char), sizeof(buff) / sizeof(char), fin);

		for (i = 0; i < count; i++)
		{
			line_length += sprintf(curr_out, "%#x, ", buff[i]);

			fprintf(fout, "%s", curr_out);
			if (line_length >= MAX_LENGTH - offset_length)
			{
				fprintf(fout, "\n%s", offset_spc);
				line_length = 0;
			}
		}
	}
	fseek(fout, -2, SEEK_CUR);
	fprintf(fout, " }");

	fclose(fout);

	return EXIT_SUCCESS;
}
