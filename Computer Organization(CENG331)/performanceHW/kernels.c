/********************************************************
 * Kernels to be optimized for the CS:APP Performance Lab
 ********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "defs.h"
/* 
 * Please fill in the following student struct 
 */
team_t team = {
    "2237899",              /* Student ID */

    "Ahmet Dara VEFA",     /* full name */
    "e223789@metu.edu.tr",  /* email address */

    "",                   /* leave blank */
    ""                    /* leave blank */
};


/***************
 * CONVOLUTION KERNEL
 ***************/

/******************************************************
 * Your different versions of the convolution functions  go here
 ******************************************************/

/* 
 * naive_conv - The naive baseline version of convolution 
 */
char naive_conv_descr[] = "naive_conv: Naive baseline implementation";
void naive_conv(int dim,int *src, int *ker,int *dst) {
    int i,j,k,l;

    for(i = 0; i < dim-8+1; i++)
        for(j = 0; j < dim-8+1; j++) {
            dst[j*dim+i] = 0;
            for(k = 0; k < 8; k++)
                for(l = 0; l < 8; l++) {
                    dst[j*dim+i] = dst[j*dim+i] +src[(j+l)*dim+(i+k)]*ker[l*dim+k];
                }
        }
            
        
}

/* 
 * convolution - Your current working version of convolution
 * IMPORTANT: This is the version you will be graded on
 */
char convolution_descr[] = "Dot product: Current working version";
void convolution(int dim,int *src, int *ker,int *dst) 
{
	#define KERDIM 8


	int *point1;//keep this for accessing elements of src
	int *dstPoint;//keep this for accesssing elements of dst
	int *kerPoint;//keep this for accessing elements of ker

	
	int result1,result2;//used for calculating the result of multiplications


	for (int i =0;i<dim*dim;++i) dst[i] = 0;

	for (int row = 0;row<dim;++row )//for each row of src
	{
		//multiply each row of ker with that row(unless it shouldn't be multiplied)
			//i.e multiply each row with [a,b] rows of ker where a is max(0,row+1-KERDIM) && b is min(row-1,7)	
		//add the result to dst[index] where index is the index of your starting element in src



		//TODO you are only calculating one rows result or something fix it 
		int dstPointOffset=row*dim;
		//if (row > dim - KERDIM+1) dstPointOffset = (dim - KERDIM+1)*dim;
		//printf("###dstPointOffset for row%d is->%d\n",row,dstPointOffset);
		dstPoint=dst+row*dim;
		//if (row > dim - KERDIM+1) dstPoint =dst+ (dim - KERDIM+1)*dim;		
		int minKerRow=0;
		int maxKerRow=row+1;
		if(KERDIM>(dim-row)) minKerRow=KERDIM-dim+row;
		if(row>=KERDIM) maxKerRow=KERDIM;
		//printf("minMaxkerRow for row%d is -> %d&&%d\n\n",row,minKerRow,maxKerRow);
		dstPointOffset -= minKerRow * dim;
		dstPoint-=minKerRow*dim;
		
		for(int kerRow=minKerRow; kerRow<maxKerRow; ++kerRow)//for each row of ker that need to multiply this row of src
		{
			point1=src+row*dim;//set point1 to start of src's row
			
			
			//printf("kerRow for row%d is -> %d\n\n",row,kerRow);
			kerPoint=ker+kerRow*dim;
			
			for (int column = 0; column < dim-7; ++column)//for each column of src until the last 8 columns
			{
			//calculate new result
				//if (dstPointOffset == 8536) printf("for row%d  && column%d  OLD result @dst%d is ->%d\n", row, column, dstPointOffset, dst[dstPointOffset]);
				result1=0;result2=0;
				result1+=kerPoint[0]*point1[0];
				result2+=kerPoint[1]*point1[1];
				result1+=kerPoint[2]*point1[2];
				result2+=kerPoint[3]*point1[3];
				result1+=kerPoint[4]*point1[4];
				result2+=kerPoint[5]*point1[5];
				result1+=kerPoint[6]*point1[6];
				result2+=kerPoint[7]*point1[7];
				//if(dstPointOffset==(dim-8)*(dim+1)) printf("for row%d  && column%d  adding new result to dst%d  ->%d\n",row,column,dstPointOffset,result1+result2);
				/*if(row==88 && column==88) printf("for row%d  && column%d  adding new result to dst%d  ->%d\n",row,column,dstPointOffset,result1+result2);
				if(row==89 && column==88) printf("for row%d  && column%d  adding new result to dst%d  ->%d\n",row,column,dstPointOffset,result1+result2);
				if(row==90 && column==88) printf("for row%d  && column%d  adding new result to dst%d  ->%d\n",row,column,dstPointOffset,result1+result2);
				if(row==91 && column==88) printf("for row%d  && column%d  adding new result to dst%d  ->%d\n",row,column,dstPointOffset,result1+result2);
				if(row==92 && column==88) printf("for row%d  && column%d  adding new result to dst%d  ->%d\n",row,column,dstPointOffset,result1+result2);
				if(row==93 && column==88) printf("for row%d  && column%d  adding new result to dst%d  ->%d\n",row,column,dstPointOffset,result1+result2);
				if(row==94 && column==88) printf("for row%d  && column%d  adding new result to dst%d  ->%d\n",row,column,dstPointOffset,result1+result2);
				if(row==95 && column==88) printf("for row%d  && column%d  adding new result to dst%d  ->%d\n",row,column,dstPointOffset,result1+result2);*/
				//save the result to this column of the row
				*dstPoint+=result1+result2;

				//select the next indexes
				++dstPoint;
				++dstPointOffset;
				++point1;

				
			}
			//if(row>87)printf("###dstPointOffset for row%d is->%d\n",row,dstPointOffset);
			dstPoint-=(2*dim-7);//bring dstPoint to the start of the last row
			dstPointOffset-=(2*dim-7);
			//if(row>87)printf("###dstPointOffset for row%d is->%d\n",row,dstPointOffset);
		}




	}
	//printf("TOTAL result for row%d  && column%d @dst%d  ->%d\n",87,87,8439,dst[8439]);
	//printf("TOTAL result for row%d  && column%d @dst%d  ->%d\n", 88, 88, 8536, dst[8536]);
	//printf("TOTAL result for row%d  && column%d @dst%d  ->%d\n", 89, 90, 8634, dst[8634]);
	//printf("TOTAL result for row%d  && column%d @dst%d  ->%d\n", 96, 97, 9313, dst[9313]);
	return;

}

/*********************************************************************
 * register_conv_functions - Register all of your different versions
 *     of the convolution functions  with the driver by calling the
 *     add_conv_function() for each test function. When you run the
 *     driver program, it will test and report the performance of each
 *     registered test function.  
 *********************************************************************/

void register_conv_functions() {
    add_conv_function(&naive_conv, naive_conv_descr);   
    add_conv_function(&convolution, convolution_descr);   
    /* ... Register additional test functions here */
}




/***************
 * MATRIX MULTIP KERNEL
 ***************/

/******************************************************
 * Your different versions of the matrix multiplications  go here
 ******************************************************/

/* 
 * naive_matrix_multiplication - The naive baseline version of matrix multiplication 
 */
char naive_matrix_multiplication_descr[] = "Naive_matrix_multiplication: Naive baseline implementation";
void naive_matrix_multiplication(int dim,int *src, int *src2,int *dst) {
    int i,j,k;

    for(i = 0; i < dim; i++)
        for(j = 0; j < dim; j++) {
            dst[j*dim+i]=0;
            for(k = 0; k < dim; k++) 
                dst[j*dim+i] = dst[j*dim+i] + src[j*dim+k]*src2[i+k*dim];
        }    
}


/* 
 * matrix_multiplication - Your current working version of matrix_multiplication
 * IMPORTANT: This is the version you will be graded on
 */
char matrix_multiplication_descr[] = "Matrix multiplications: Current working version";
void matrix_multiplication(int dim,int *src, int *src2,int *dst) 
{


    //naive_matrix_multiplication(dim,src,src2,dst);

	//############TODO instead of going through row by column go with element by row, this way less memory is accessed
	//############TODO complete unrolling with templates? does ansi c support it?
	
	//TODO reverse iteration?
	//TODO since dimensions are multiple of 2 find log2 of dim and 
		//change multiply/divison operations to << & >>
	//TODO instead of row++ do ++row
	//TODO find the fastest for loop condition check mechanism(i.e row<dim? row!=dim?)
	//TODO unroll loops(last)






	/*for (int columnSrc = 0; columnSrc < dim; ++columnSrc)//for each column of src
	{
		for (int rowSrc = 0; rowSrc < dim; ++rowSrc)//for each row of src
		{
			//multiply src[rowSrc,columnSrc] with src2[columnSrc,columnSrc2](columnSrc'th row of src2)
			//then add it to dst[rowSrc,columnSrc2]
			int rowSrcDim=rowSrc*dim;
			int columnSrcDim=columnSrc*dim;//TODO check effect of order on performance(pipeline)
			int srcValue=src[rowSrcDim+columnSrc];

			for (int columnSrc2 = 0; columnSrc2 < dim; ++columnSrc2)//for each column in that particular row of src2
			{
				dst[rowSrcDim+columnSrc2]+=srcValue *src2[columnSrcDim+columnSrc2];
			}
		}
	}*/


	
	// //transpose using blocks with size blocksize
	// int blocksize=32;
	// int kDim;
	// for (int i = 0; i < dim; i += blocksize) 
	// {
 //    	for (int j = 0; j < dim; j += blocksize) 
 //    	{
 //        // transpose the block beginning at [i,j]
 //        	for (int k = i; k < i + blocksize; ++k) 
 //        	{
 //        		kDim=k*dim;
 //            	for (int l = j; l < j + blocksize; ++l) 
 //            	{
 //                	transMat[k + l*dim] = src2[l + kDim];
 //            	}
 //        	}
 //    	}
	// }


	//TRANSPOSE src2 save it to transMat
	//multiply each row of src with transMat
	//save the result to dst
	#define intSize sizeof(int)
	int dimsqrMinus1=dim*dim-1;
	int *transMat=(int*)malloc(intSize*(dimsqrMinus1+1));

	int *transPoint=transMat,*point2=src2;
	
	for (int row = dim; row --;)//for each row of src2
	{
		
		for (int column = dim>>5; column--;)//for each column of src2
		{
			//set transmat[column*dim+row]=src2[row*dim+column]

			(*transPoint)=(*point2);
			transPoint[dim]=point2[1];//unroll
			transPoint[2*dim]=point2[2];
			transPoint[3*dim]=point2[3];
			transPoint[4*dim]=point2[4];
			transPoint[5*dim]=point2[5];
			transPoint[6*dim]=point2[6];
			transPoint[7*dim]=point2[7];
			transPoint[8*dim]=point2[8];
			transPoint[9*dim]=point2[9];
			transPoint[10*dim]=point2[10];
			transPoint[11*dim]=point2[11];
			transPoint[12*dim]=point2[12];
			transPoint[13*dim]=point2[13];
			transPoint[14*dim]=point2[14];
			transPoint[15*dim]=point2[15];
			transPoint[16*dim]=point2[16];
			transPoint[17*dim]=point2[17];
			transPoint[18*dim]=point2[18];
			transPoint[19*dim]=point2[19];
			transPoint[20*dim]=point2[20];
			transPoint[21*dim]=point2[21];
			transPoint[22*dim]=point2[22];
			transPoint[23*dim]=point2[23];
			transPoint[24*dim]=point2[24];
			transPoint[25*dim]=point2[25];
			transPoint[26*dim]=point2[26];
			transPoint[27*dim]=point2[27];
			transPoint[28*dim]=point2[28];
			transPoint[29*dim]=point2[29];
			transPoint[30*dim]=point2[30];
			transPoint[31*dim]=point2[31];

			point2+=32;
			transPoint+=32*dim;
		}
		transPoint-=dimsqrMinus1;//transPoint++;
	}


/*
	int iDim;
	for (int i = 0; i != dim; ++i)//transposing src2
	{
		iDim=i*dim;
		for (int j = 0; j != dim; ++j)
		{
			transMat[j*dim+i]=src2[iDim+j];		
		}
	}
*/

	register int sum=0,sum2=0;
	register int *point1=src,*dstPoint=dst,*point1beforeTraversingColumn=src;
	for (int row = dim; row--;)//for each row of src
	{
		point2=transMat;//set|reset point2 to beginning of transMat;
		for (int row2 = dim; row2--;)//for each row of transMat
		{
			sum=0;
			sum2=0;
			

			//set start of the row for point1
			point1beforeTraversingColumn=point1;

			for (int column = dim>>5; column--;)//for each column of src or transMat
			{
				//multiply src[row,column] with transMat[row2,column], then save the SUM to dst[row,row2]
				sum+=(*point1)*(*point2); //sum+=src[rowDim+column]*transMat[row2Dim+column];
				sum2+=(point1[1])*(point2[1]);
				sum+=(point1[2])*(point2[2]);
				sum2+=(point1[3])*(point2[3]);
				sum+=(point1[4])*(point2[4]);
				sum2+=(point1[5])*(point2[5]);
				sum+=(point1[6])*(point2[6]);
				sum2+=(point1[7])*(point2[7]);
				sum+=(point1[8])*(point2[8]);
				sum2+=(point1[9])*(point2[9]);
				sum+=(point1[10])*(point2[10]);
				sum2+=(point1[11])*(point2[11]);
				sum+=(point1[12])*(point2[12]);
				sum2+=(point1[13])*(point2[13]);
				sum+=(point1[14])*(point2[14]);
				sum2+=(point1[15])*(point2[15]);
				sum+=(point1[16])*(point2[16]);
				sum2+=(point1[17])*(point2[17]);
				sum+=(point1[18])*(point2[18]);
				sum2+=(point1[19])*(point2[19]);
				sum+=(point1[20])*(point2[20]);
				sum2+=(point1[21])*(point2[21]);
				sum+=(point1[22])*(point2[22]);
				sum2+=(point1[23])*(point2[23]);
				sum+=(point1[24])*(point2[24]);
				sum2+=(point1[25])*(point2[25]);
				sum+=(point1[26])*(point2[26]);
				sum2+=(point1[27])*(point2[27]);
				sum+=(point1[28])*(point2[28]);
				sum2+=(point1[29])*(point2[29]);
				sum+=(point1[30])*(point2[30]);
				sum2+=(point1[31])*(point2[31]);

				point1+=32;//get to the next 32 position in the row or start of the next row
				point2+=32;//get to the next 32 position in the row or start of the next row
			}

			//reset p1 to the start of the same row && dont touch p2 as it is incremeneted in the for loop
			point1=point1beforeTraversingColumn;



			*dstPoint=sum+sum2;//set dst's value
			dstPoint++;//get to the next column or start of next row



			////point2+=dim; //row2Dim=row2*dim;
		}
		point1+=dim;//get point1 to the start of next row//rowDim=row*dim;
		
	}



	free(transMat);
	return;




	// for (int row = 0; row < dim; row++)//for each row of src
	// {
	// 	//multiply that row with each column of src2
	// 	//then add it to dst[row,column]=dst[row*dim+column]

	// 	/*elements to multiply are:
	// 	*	src[row,k]=src[row*dim+k]
	// 	*	src2[k,column]=src2[k*dim+column]
	// 	*/
	// 	int rowDim = row * dim;
	// 	for (int column = 0; column < dim; column++)//for each column
	// 	{
	// 		dst[rowDim + column] = 0;//set to 0 just in case// this cache's dst[rowDim+column]
	// 		for (int k = 0; k < dim; k++)
	// 		{
	// 			//TODO check access patterns/coalescing/strided access etc.
	// 			dst[rowDim + column] += src[rowDim + k] * src2[k*dim + column];//src has coalesced access, src2 has strided access
	// 		}
	// 	}
	// }

}

/*********************************************************************
 * register_matrix_multiplication_functions - Register all of your different versions
 *     of the matrix multiplication  with the driver by calling the
 *     add_matrix_multiplication_function() for each test function. When you run the
 *     driver program, it will test and report the performance of each
 *     registered test function.  
 *********************************************************************/

void register_matrix_multiplication_functions() {
    //add_matrix_multiplication_function(&naive_matrix_multiplication, naive_matrix_multiplication_descr);   
    //add_matrix_multiplication_function(&matrix_multiplication, matrix_multiplication_descr);   
    /* ... Register additional test functions here */
}
