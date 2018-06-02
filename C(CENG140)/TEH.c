#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
/* test input 3+4+$(57.3) */
char stack[200];
double finalformula[200];
int stacktop=0,finalformulatop=0;
int countofintervals;
long int countofexps;
int countofvariables=0,fakecountofvariables=0;
double variablesarray[30][105];
int columnnumber=0;
char probabilityarray[30][1000];
double intervallimits[30][105];
/*'+','-','*','/','^','@','$','~','%','&'
  43  45  42  47  94  64  36  126 37  38   */

/* operator presedence's // @=cos // $=sin // %=sqrt // &= ln */
int priority(char operator)
{
  switch(operator)
  {
    case '(': return 0;
    case '+':
    case '-': return 1;
    case '*':
    case '/': return 2;
    case '^': return 3;
    case '@':
    case '$':
    case '~':
    case '%':
    case '&': return 4;
  }
  return 0;
}


int formulconverter(char *formul)
{
  int i=0;
  while(formul[i]!='\n')
  {
    if (isdigit(formul[i]))
    {
      int j;
      /* for loop is to increment the index of char stack to the next operator*/
      for(j=0;isdigit(formul[i+j]) || formul[i+j]=='.';)
      {
        j++;
      }
      finalformula[finalformulatop]=atof(formul+i);
      finalformulatop++;
      i+=j-1;
    }
    else if(isalnum(formul[i]))
    {
      finalformula[finalformulatop]=(double)(-(formul[i]));
      finalformulatop++;
    }
    else if(formul[i]=='(')
    {
      stack[stacktop]=formul[i];
      stacktop++;
    }
    else if(formul[i]==')')
    {
      while(stack[stacktop-1]!='(')
      {
        finalformula[finalformulatop]=(double)(-(stack[stacktop-1]));
        finalformulatop++;
        stacktop--;
      }
      stacktop--;
    }
    /* taking operands to finalformula or stack */
    else
    {
      while(stacktop>0 && priority(stack[stacktop-1])>=priority(formul[i]))
      {
        finalformula[finalformulatop]=(double)(-(stack[stacktop-1]));
        finalformulatop++;
        stacktop--;
      }
      stack[stacktop]=formul[i];
      stacktop++;
    }
/* the finalformula might be fucked up ex: difference between 65.0 from 'A' DONE? */
    i++;
  }
  while(stacktop!=0)
  {
    finalformula[finalformulatop]=(double)(-(stack[stacktop-1]));
    finalformulatop++;
    stacktop--;
  }
  return 0;
}

void intervallimiter(int arraycount,double elements[],int intervalcount)
{
int k=0,k2=0;
  while(k<arraycount)
  {
    intervallimits[k][0]=elements[0];
    for(k2=1;k2<intervalcount+3;k2++)
    {
      intervallimits[k][k2]=elements[1]+((elements[2]-elements[1])*(k2-1)/intervalcount);
    }
    k++;
  }
}






int main()

{
  char formul[201];
  int i=0,t=0,t2=0,elementnumber=0;
  float decoyvariables[200];
  char decoyelement;
  int totalindex=0,r=0,q=0,q2=0;
  /* take the formul*/
  while(i<200 )
  {
    scanf("%c",(formul+i));
    if(i!=0 && *(formul+i)=='\n' )
    {
      break;
    }
    /* eleminate the empty chars*/
    if(*(formul+i)==' ')
    {
      i--;
    }
    /* converting cos and shyte to @ and shyte */
    else if(*(formul+i)=='s')
    {
      scanf("%c",(formul+i));
      if(*(formul+i)=='i')
      {
        scanf("%c",(formul+i));
        *(formul+i)='$';
      }
      else
      {
        scanf("%c",(formul+i));
        scanf("%c",(formul+i));
        *(formul+i)='%';
      }
    }

    else if(*(formul+i)=='c')
    {
      scanf("%c",(formul+i));
      scanf("%c",(formul+i));
      *(formul+i)='@';
    }

    else if(*(formul+i)=='l')
    {
      scanf("%c",(formul+i));
      *(formul+i)='&';
    }

    i++;
  }
  /* i is number of elements in formul so MAX INDEX is i-1 */
  /* convert sin and shyte to $ and shyte */


  /* converting the formul to postfix*/
  formulconverter(formul);
/*to print the POSTFİX
  for(a=0;a<finalformulatop;a++)
  {
    printf(" %f ",finalformula[a]);
  }
*/
/*get the count of exp and interval */
scanf(" %d %ld",&countofintervals,&countofexps);


/* calculate number of variables */
for(t=0,t2=0;t<finalformulatop;t++)
{
  if(-90<=finalformula[t] && finalformula[t]<=-65 )
  {
    fakecountofvariables++;
    decoyvariables[t2]=finalformula[t];
    t2++;
  }
}
countofvariables=fakecountofvariables;
printf("COV%d",countofvariables);
for(t=0;t<fakecountofvariables-1;t++)
{
  if(decoyvariables[t])
  {
    printf("entered");
    for(t2=t+1;t2<fakecountofvariables;t2++)
    {

      if(decoyvariables[t]==decoyvariables[t2])
      {

        countofvariables--;
        decoyvariables[t2]=0;
      }
    }
  }
}
printf("HERE%d",countofvariables);


/* take interval probabilities for each variable */
while(elementnumber<countofvariables)
{
  scanf("\n %c",&decoyelement);
  variablesarray[elementnumber][0]=(double)(decoyelement);
  /*
  printf(" VAL%f  %d \n",variablesarray[elementnumber][0],countofintervals); */
  while(columnnumber<countofintervals+2)
  {
    columnnumber++;
    scanf("%lf",&(variablesarray[elementnumber][columnnumber]));
    /*
    printf("%f\n",variablesarray[elementnumber][columnnumber]); */

  }
  columnnumber=0;
  elementnumber++;
}

/* make a 1000 char array for each variable and store probability value(0-1...countofintervals) */
for(q=0;q<countofvariables;q++)
{
  totalindex=0;
  r=0;
  for(q2=0;q2<countofintervals;q2++)
  {
    totalindex+=(variablesarray[q][3+q2])*1000;
    while(r<totalindex)
    {
      /* first interval is 0 last one is countofintervals */
      probabilityarray[q][r]=q2;
      r++;
    }
  }
}


/* get float arrays to indicate interval limits */
/* MIGHT BE FUCKED UP */
intervallimiter(countofvariables,variablesarray[0],countofintervals);


  return 0;
}