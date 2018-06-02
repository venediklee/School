#include <stdio.h>
#include <math.h>

int main()
{
	char operator,first;
	int sum1=0,sum2=0;
	float result;

	scanf(" %c",&operator);

	while(1)
	{
		scanf(" %c",&first);
		if(first=='#') break;
		else if(first=='1') sum1=sum1*2+1;
		else if(first=='0') sum1*=2;
		else continue;
	}
	if (operator=='A')
	{
		while(1)
		{
		scanf(" %c",&first);
		if(first=='#') break;
		else if(first=='1') sum2=sum2*2+1;
		else if(first=='0') sum2*=2;
		else continue;
		}
	}
	switch(operator)
	{
		case 'A':result=sum1+sum2;printf("%.2f",result);break;
		case 'T':result=log10(sum1);printf("%.2f",result);break;
	}
	return 0;
}