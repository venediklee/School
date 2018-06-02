`timescale 1ns / 1ps

module lab4_2(
	input[3:0] userID, 
	input CLK, 
	input team,
	input [1:0] mode,
	output reg  [7:0] numPlyLP,
	output reg  [7:0] numPlyCF,
	output reg  teamWng,
	output reg capWngLP,
	output reg capWngCF,
	output reg findRes,
	output reg [3:0] listOut,
	output reg  listMode
    );
//Write your code below

	reg [3:0]CengForce[4:0] ; 
	reg [3:0]LogicPower[4:0] ;
	integer i, flag = 0, listCtr = 0, flag2, flag3 ; 

initial 
	begin 
	numPlyLP = 0 ; 
	numPlyCF = 0 ;
	capWngCF = 1 ;
	capWngLP = 1 ;
	for(i = 0 ; i < 5 ; i = i + 1) 
		begin
			CengForce[i] = 4'b0000 ; 
			LogicPower[i] = 4'b0000 ; 
		end
	end
			
			
always@(posedge CLK)
	begin
		flag = 0 ;
		//findRes = 0 ;
		
		if(mode == 2'b01) // LOGIN MODE 
			begin
			listMode = 0 ;
			listCtr = 0 ; 
				if(userID[3] == team) teamWng = 0 ; //CHECKS IF THE ID MATCHES THE TEAM YOU WANT TO LOG INTO
				else teamWng = 1 ; 
				
				if(!teamWng && team && numPlyCF < 5) //PLAYER WANTS TO LOG INTO CF, CF HAS ROOM
					begin
						for(i = 0 ; i < 5 ; i = i + 1) // CHECK IF THE SAME ID IS IN CF
							begin
								if (CengForce[i] == userID) flag = 1 ;
							end
						if(!flag) // PLAYER JOINS CF 
							begin
								for (i = 0 ; i < 5 ; i = i + 1) 
									begin
										if(flag2 && 	CengForce[i] == 4'b0000)
											begin 
												CengForce[i] = userID ;
												flag2 = 0 ;
											end
									end  
								numPlyCF = numPlyCF + 1 ; 
								flag = 0 ;
								flag2 = 1 ; 
							end 
					end
					
					else if(!teamWng && !team && numPlyLP < 5) //PLAYER WANTS TO LOG INTO LP, LP HAS ROOM
					begin
						for(i = 0 ; i < 5 ; i = i + 1) // CHECK IF THE SAME ID IS IN LP
							begin
								if (LogicPower[i] == userID) flag = 1 ; 
							end
						if(!flag) // PLAYER JOINS LP 
							begin
								for (i = 0 ; i < 5 ; i = i + 1) 
									begin
										if(flag2 && LogicPower[i] == 4'b0000) 
										begin
											LogicPower[i] = userID ;
											flag2 = 0 ;
										end
									end 
									
								numPlyLP = numPlyLP + 1 ; 
								flag = 0 ;
								flag2 = 1 ;
							end 
					end
			end
			
		else if(mode == 2'b00) //LOGOUT
			begin
			listMode = 0 ;
			listCtr = 0 ; 
				if(userID != 4'b0000)
					begin
						for(i = 0 ; i < 5 ; i = i + 1) 
							begin
								if (flag2 && LogicPower[i] == userID)
									begin
										LogicPower[i] = 4'b0000 ;
										numPlyLP = numPlyLP - 1 ;
										flag2 = 0 ; 
									end
								if (flag2 && CengForce[i] == userID)
									begin
										CengForce[i] = 4'b0000 ;
										numPlyCF = numPlyCF - 1 ; 
										flag2 = 0 ; 
									end
								end
								flag2 = 1 ; 
						end
			end
		else if (mode == 2'b10) // FIND MODE 
			begin
			listMode = 0 ;
			listCtr = 0 ; 
			flag3 = 0 ; 
				if(team)
					begin 
						for (i = 0 ; i < 5 ; i = i + 1)
							begin
								if(CengForce[i] == userID) 
									begin
										findRes = 1 ; 
										flag3 = 1 ;
									end
							end		
					end
				else
					begin 
						for (i = 0 ; i < 5 ; i = i + 1)
							begin
								if(LogicPower[i] == userID) 
								begin 
									findRes = 1 ; 
									flag3 = 1 ;
								end
							end
					end
				if(!flag3) findRes = 0 ; 

			end 
		else // LIST MODE  
			begin
				listMode = 1 ; 
				if (team) 
					begin 
						listOut = CengForce[listCtr] ; 
						listCtr = (listCtr + 1) ;
						if(listCtr == 5) listCtr = 0 ; 
					end
				else 
					begin
						listOut = LogicPower[listCtr] ; 
						listCtr = (listCtr + 1) ;
						if(listCtr == 5) listCtr = 0 ; 
					end 
			end
			
		if(numPlyLP == 0 || numPlyLP == 5) capWngLP = 1 ; //CHECKS IF THE TEAMS ARE FULL OR EMPTY
		else capWngLP =0; 
		if(numPlyCF == 0 || numPlyCF == 5) capWngCF = 1 ; // SAME AS ABOVE 
		else capWngCF = 0;
	end

endmodule
