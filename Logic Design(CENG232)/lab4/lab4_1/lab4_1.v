`timescale 1ns / 1ps
module FUNCROM (input [3:0] romAddress, output reg[4:0] romData);


integer i=0;
initial 
begin
	for(i=0;i<=4;i=i+1) 
	begin
		romData[i]=0;
	end
end

always@(romAddress)
begin
	case(romAddress)
		4'b0000 : romData = 5'b00000;
		4'b0001 : romData = 5'b00010;
		4'b0010 : romData = 5'b00100;
		4'b0011 : romData = 5'b00111;
		4'b0100 : romData = 5'b01010;
		4'b0101 : romData = 5'b01011;
		4'b0110 : romData = 5'b01101;
		4'b0111 : romData = 5'b01110;
		4'b1000 : romData = 5'b10001;
		4'b1001 : romData = 5'b10010;
		4'b1010 : romData = 5'b10100;
		4'b1011 : romData = 5'b10111;
		4'b1100 : romData = 5'b11001;
		4'b1101 : romData = 5'b11010;
		4'b1110 : romData = 5'b11110;
		4'b1111 : romData = 5'b11111;
	endcase
/*
	romData[0]=romAddress[0];
	romData[1]=romAddress[1];
	romData[2]=romAddress[2];
	romData[3]=(romAddress[3]+(~romAddress[0]*romAddress[1]*(~romAddress[2]))+(romAddress[0]*romAddress[1]*romAddress[2]))%2;
	romData[4]=(((~romAddress[1])*romAddress[2]*romAddress[3])+(romAddress[0]*(romAddress[2]~^romAddress[3]))+((~romAddress[0])*romAddress[1]*((~romAddress[2])~^romAddress[3])))%2;
*/
end
/*Write your code here*/
endmodule
																						
module FUNCRAM (input mode,input [3:0] ramAddress, input [4:0] dataIn,input op, input [1:0] arg,  input CLK, output reg [8:0] dataOut);
/*Write your code here*/

reg [8:0] memo [15:0]; 
integer memoAddress;
integer i;
initial
begin
	for(i=0;i<=15;i=i+1)
	begin
		memo[i]=9'b000000000;
	end
end

//read executions

always@(mode,ramAddress,dataIn,op,arg)
begin
	if(mode==0)
	begin
		memoAddress=ramAddress;
		dataOut=memo[memoAddress];
	end
end




integer argDecimal;
integer totalNum;
integer outputIndex;
integer trickInt;
//write executions
always@(posedge CLK)
begin
//put the arg value as decimal to an integer
	argDecimal=0;
	totalNum=0;
	case({arg[1],arg[0]})
		2'b0_0 : argDecimal = 2 ;
		2'b0_1 : argDecimal = 1 ;
		2'b1_0 : argDecimal = -1 ;
		2'b1_1 : argDecimal = -2 ;
	endcase
	

	if(mode==1)
	begin
		
	
	
	//derivative executions
		//4x^3 3x^2 2x 1
			//+ or -
				//choose between + or - with dataIn variable >>>>>>>>>> i.e 0 is +     1 is -
	if(op==1)
	begin
		totalNum=0;
		if(dataIn[4]==0)
		begin
			totalNum=totalNum+4*(argDecimal*argDecimal*argDecimal);
		end
		else
		begin
			totalNum=totalNum-4*(argDecimal*argDecimal*argDecimal);
		end
		
		if(dataIn[3]==0)
		begin
			totalNum=totalNum+3*(argDecimal*argDecimal);
		end
		else
		begin
			totalNum=totalNum-3*(argDecimal*argDecimal);
		end
		
		if(dataIn[2]==0)
		begin
			totalNum=totalNum+2*(argDecimal);
		end
		else
		begin
			totalNum=totalNum-2*(argDecimal);
		end
		
		if(dataIn[1]==0)
		begin
			totalNum=totalNum+1;
		end
		else
		begin
			totalNum=totalNum-1;
		end
		
		
		//do the modulo by 2 trick(put the first output to the dataOut[8])
			//dataOut[0] is decided by the sign of the argDecimal
		trickInt=(totalNum>=0)? totalNum: -totalNum;
		memoAddress=ramAddress;
		for(outputIndex=8; outputIndex>0;outputIndex=outputIndex-1)
		begin
			memo[memoAddress][8-outputIndex]=(trickInt%2);
			trickInt=trickInt/2;
		end
		memo[memoAddress][8]= (totalNum>=0)?0 : 1;
		
		
	end
	
	
	//modulo 7 executions
		//x^4 x^3 x^2 x 1
			//+ or -
				//choose between + or - with dataIn variable >>>>>>>>>> i.e 0 is +     1 is -
	else
	begin
		totalNum=0;
		if(dataIn[4]==0)
		begin
			totalNum=totalNum+(argDecimal*argDecimal*argDecimal*argDecimal);
		end
		else
		begin
			totalNum=totalNum-(argDecimal*argDecimal*argDecimal*argDecimal);
		end
		
		if(dataIn[3]==0)
		begin
			totalNum=totalNum+(argDecimal*argDecimal*argDecimal);
		end
		else
		begin
			totalNum=totalNum-(argDecimal*argDecimal*argDecimal);
		end
		
		if(dataIn[2]==0)
		begin
			totalNum=totalNum+(argDecimal*argDecimal);
		end
		else
		begin
			totalNum=totalNum-(argDecimal*argDecimal);
		end
		
		if(dataIn[1]==0)
		begin
			totalNum=totalNum+argDecimal;
		end
		else
		begin
			totalNum=totalNum-argDecimal;
		end
		
		if(dataIn[0]==0)
		begin
			totalNum=totalNum+1;
		end
		else
		begin
			totalNum=totalNum-1;
		end
		
		//now get the %7 of totalNum which is a decimal
			//then put it to the required position which is taken from 
		
		//totalNum=totalNum%7;
		for(i=10; i>0; i=i-1)
		begin
			if(totalNum>7)
			begin
				totalNum=totalNum-7;
			end
		end
		trickInt=(totalNum>=0)? totalNum: -totalNum;
		memoAddress=ramAddress;
		for(outputIndex=8; outputIndex>0;outputIndex=outputIndex-1)
		begin
			memo[memoAddress][8-outputIndex]=(trickInt%2);
			trickInt=trickInt/2;
		end
		memo[memoAddress][8]= (totalNum>=0)?0 : 1;
		
		end
	end
	
	
end




	
endmodule


module FUNCMEMORY(input mode, input [6:0] memInput, input CLK, output wire [8:0] result);
	/*Don't edit this module*/
	wire [4:0]  romData;

	FUNCROM RO(memInput[6:3], romData);
	FUNCRAM RA(mode, memInput[6:3], romData, memInput[2],memInput[1:0], CLK, result);

endmodule