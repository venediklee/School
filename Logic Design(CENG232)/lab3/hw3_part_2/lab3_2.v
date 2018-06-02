`timescale 1ns / 1ps 
module lab3_2(
			input[4:0] word,
			input CLK, 
			input selection,
			input mode,
			output reg [7:0] hipsterians1,
			output reg [7:0] hipsterians0,
			output reg [7:0] nerdians1,
			output reg [7:0] nerdians0,
			output reg warning
    );

	initial begin
		hipsterians0=0;
		nerdians0=0;
		hipsterians1=0;
		nerdians1=0;
		warning=0;
	end
   //Modify the lines below to implement your design .
	always@(posedge CLK) begin
		for(i=0;i<4;i=i+1) begin
			if(word[i]==word[i+1] && word[i]==selection)
					if(mode==0) begin
							////////            DECREASE the value
							if(selection==0) begin
								// decrease the value of hipsterians
								end
							else begin 
								// decrease the value of nerdians
								end
							end
					else begin
							////////            INCREASE the values
							if(selection==0) begin
								//increase the value of hipsterians
								end
							else begin
								//increase the value of nerdians
								end
							end
			else if(i==3) begin
					//give error && dont change any values
					end
			end
	end
endmodule


