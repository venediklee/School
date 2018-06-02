module Hw1 where

data Cell = H | P | O | X deriving (Read,Show,Eq)
data Result = Fail | Caught (Int,Int) deriving (Read,Show,Eq)
data Direction = N | S | E | W deriving (Read,Show,Eq)

simulate :: [[Cell]] -> [(Direction, Direction)] -> Result
-- DO NOT CHANGE ABOVE THIS LINE, WRITE YOUR CODE BELOW --

--general idea:: add locations of Hunter Prey and Obstacles to a list
    --also add size of the map to the list
		--do checks//update that list etc
--make the list (######my function for updating is done at otherwise)
simulate coords directs
	| null coords = Fail
	| hPos == pPos = Caught hPos
	| null directs = Fail
	| otherwise = compute [x | x<-theList, fst x >=0, snd x>=0] directs (length (coords!!0)) (length coords)
	where hPos = findPos H coords 0 0
	      pPos = findPos P coords 0 0
	      theList=[hPos]++([pPos]++(listAdder coords 0))

--find the main list where the element resides
findPos :: Cell -> [[Cell]] -> Int -> Int -> (Int, Int)
findPos element coords2 x y
	| null coords2 || (y> (length coords2) -1) = (-10,-10)
	| element `elem` (coords2 !! y) = ( findPos2 element (coords2 !! y) 0 0, y )
	| otherwise = findPos element coords2 x (y+1)

--find the position of the element in the miniList
findPos2 :: Cell -> [Cell] -> Int -> Int -> Int 
findPos2 element2 miniCoords x2 repeatNumber
	| null miniCoords = -10
	| (x2 > (length miniCoords) -1) = -10
	| element2 == (miniCoords !! x2) && repeatNumber==0 = x2
	| element2 == (miniCoords !! x2) && repeatNumber>0 = findPos2 element2 miniCoords (x2+1) (repeatNumber-1)
	| otherwise = findPos2 element2 miniCoords (x2+1) repeatNumber

--adds X's to theList
listAdder::[[Cell]]-> Int->[(Int,Int)]
listAdder coordsX rowNumber
	|null coordsX = []
	|rowNumber> ((length coordsX) -1)= []
	|X `elem` (coordsX !! rowNumber) = (listAdder2 rowNumber (coordsX!!rowNumber) 0) ++ (listAdder coordsX (rowNumber+1))
	|otherwise = listAdder coordsX (rowNumber+1) 
	--where (_,restOfTheList)=splitAt (findPos2 X (coordsX!!rowNumber) 

--numBeforeX is for finding the next X with findPos2, rowNumber2 is for the y part of the tuple,
	-- miniCoordsX is the list where we search
listAdder2::Int->[Cell]->Int->[(Int,Int)]
listAdder2 rowNumber2 miniCoordsX numOfBeforeX
	|null miniCoordsX =[(-10,-10)]
	|numOfBeforeX>((length miniCoordsX)-1) = []
	|otherwise = [(findPos2 X miniCoordsX 0 numOfBeforeX,rowNumber2)]++(listAdder2 rowNumber2 miniCoordsX (numOfBeforeX+1))

--hunter and prey are positions, hunterNext and preyNext are next positions
--computes the outcome
compute::[(Int,Int)] -> [(Direction,Direction)] ->Int -> Int -> Result
compute listofPositions directions xLength yLength
	|length listofPositions<2 = Fail
	|(hunter)==(prey) = Caught (hunter)
	|length directions<1 = Fail
	|otherwise = compute (hunterNext:preyNext :(tail (tail listofPositions))) (tail directions) xLength yLength
	where hunter = (listofPositions!!0)
	      prey = (listofPositions!!1)
	      hunterNext = positionChanger hunter (fst (directions!!0)) xLength yLength (tail (tail listofPositions))
	      preyNext = positionChanger prey (snd (directions!!0)) xLength yLength (tail (tail listofPositions))

--waypoint is NESW, xLength2 and yLength2 are size of the playfield, xlist is list of X coordinates
--gives the next position
    --also prevents going outside of the boundries and inside of X's
positionChanger:: (Int,Int)-> Direction -> Int -> Int -> [(Int,Int)] -> (Int,Int)
positionChanger currentPosition waypoint xLength2 yLength2 xList
	|waypoint==N = if(y3==0 || (x3, y3-1) `elem` xList) then currentPosition else ( x3, y3 -1)
	|waypoint==S = if(y3==yLength2-1 || (x3,y3+1) `elem` xList) then currentPosition else ( x3, y3 +1)
	|waypoint==W = if(x3==0 || (x3-1,y3) `elem` xList) then currentPosition else ( x3 -1, y3)
	|waypoint==E = if(x3==xLength2-1 || (x3+1,y3) `elem` xList) then currentPosition else ( x3 +1, y3)
	where (x3,y3)= currentPosition


	-----------i need to do the movement calculations now (in compute)
	-- the otherwise part is for error checking right now 
	--you may need to add extra checks xd