module Hw2 where

import Data.List -- YOU MAY USE THIS MODULE FOR SORTING THE AGENTS

data Level = Newbie | Intermediate | Expert deriving (Enum, Eq, Ord, Show, Read)
data Hunter = Hunter {hID::Int, hlevel::Level, hEnergy::Int, hNumberOfCatches::Int, hActions::[Direction]} deriving (Eq, Show, Read)
data Prey = Prey {pID::Int, pEnergy::Int, pActions::[Direction]} deriving (Eq, Show, Read)
data Cell = O | X | H Hunter | P Prey | T deriving (Eq, Show, Read)
data Direction = N | S | E | W deriving (Eq, Show, Read)
type Coordinate = (Int, Int)
-- DO NOT CHANGE THE DEFINITIONS ABOVE. --


-- INSTANCES OF Ord FOR SORTING, UNCOMMENT AND COMPLETE THE IMPLEMENTATIONS --
instance Ord Hunter where
	compare h1 h2
		|(hlevel h1)>(hlevel h2)=GT
		|(hlevel h1)==(hlevel h2) && (hEnergy h1)>(hEnergy h2)=GT
		|(hlevel h1)==(hlevel h2) && (hEnergy h1)==(hEnergy h2) && (hNumberOfCatches h1)>(hNumberOfCatches h2)=GT
		|(hlevel h1)==(hlevel h2) && (hEnergy h1)==(hEnergy h2) && (hNumberOfCatches h1)==(hNumberOfCatches h2) && (hID h1)<(hID h2)=GT
		|otherwise = LT


instance Ord Prey where
	compare p1 p2
		|(pEnergy p1)> (pEnergy p2) = GT
		|(pEnergy p1)==(pEnergy p2) && (pID p1)< (pID p2)= GT
		|otherwise = LT 


-- WRITE THE REST OF YOUR CODE HERE --


--simulates the output
simulate ::[ [ Cell ] ] -> ( [ ( Hunter , Coordinate ) ] , [ ( Prey , Coordinate ) ] )
simulate coordinates
	|null coordinates= ([],[])
	|otherwise =  ([],[])  --compute ([h | h<-hPoses, fst (snd h) >=0, snd (snd h) >=0]) ([hd | hd<-hDirs, hd/=[]]) ([p | p<-pPoses, fst (snd p) >=0, snd (snd p) >=0]) ([pd | pd<-pDirs, pd/=[]]) ([x | x<-xPoses, fst x >=0, snd x >=0]) ([t | t<-tPoses, fst t >=0, snd t >=0]) (length (coordinates!!0)) (length coordinates)
	where (hPoses,hDirs)=posFinderh hunter coordinates
	      (pPoses,pDirs)=posFinderp prey coordinates
	      (xPoses,_)=posFinder X coordinates
	      (tPoses,_)=posFinder T coordinates
	      hunter= H {}
	      prey= P {}


--takes hunters' positions(and themselves) and directions, preys' positions(and themselves) and directions,
	-- takes X positions and T positions, and boundries
--returns the result
compute::[(Hunter,Coordinate)] -> [[Direction]] ->[(Prey,Coordinate)]-> [[Direction]]-> [Coordinate]->[Coordinate]->Int->Int->( [ ( Hunter , Coordinate ) ] , [ ( Prey , Coordinate ) ] )
compute hPos hDir pPos pDir xPos tPos xBound yBound
	|null hDir || null (hDir!!0) = (hPos,pPos)
	|otherwise = ([],[])
	


-------------------------------------

--find the position of the given element(and sort them if they are P or H)
	--first list is the location of the element(sorted), second list is movements
posFinder:: Cell -> [[Cell]] -> ([Coordinate],[[Direction]])
posFinder cell  coords= (listAdder coords 0 cell,[[]])
posFinderh:: Cell -> [[Cell]] -> ([(Hunter,Coordinate)],[[Direction]] )
posFinderh H{}  coords
	|null coords=([],[])
	|otherwise= (hlocationlist,directionAdder coords H{} [snd hl | hl<-hlocationlist] 0)
	where hlocationlist=posFixerh (listAdder coords 0 H{}) coords
posFinderp:: Cell -> [[Cell]] -> ([(Prey,Coordinate)],[[Direction]])
posFinderp P{}  coords
	|null coords=([],[])
	|otherwise=(plocationlist,directionAdder coords P{} [ snd pl | pl<-plocationlist] 0)
	where plocationlist=posFixerp (listAdder coords 0 P{}) coords


--hunters//preys are in the fst extraList2
--positions are in the snd extraList2
posFixerh::[Coordinate]-> [[Cell]]->[(Hunter,Coordinate)]
posFixerh posList coordinates2
	|null posList || null coordinates2= []
	|otherwise= extraList2
	where extraList=posFixer2h posList coordinates2 0
	      extraList2= sortBy (\(a,_) (b,_) -> compare a b) extraList
posFixerp::[Coordinate]-> [[Cell]]->[(Prey,Coordinate)]
posFixerp posList coordinates2
	|null posList || null coordinates2= []
	|otherwise= extraList2
	where extraList=posFixer2p posList coordinates2 0
	      extraList2= sortBy (\(a,_) (b,_) -> compare a b) extraList


posFixer2h::[Coordinate] ->[[Cell]]->Int ->[(Hunter,Coordinate)]
posFixer2h posList2 coordinates3 index2
	|null posList2 || null coordinates3 || index2>((length coordinates3)-1)= []
	|otherwise= [(function_h((coordinates3!!(snd coordinates4))!!(fst coordinates4)),coordinates4)]++ posFixer2h posList2 coordinates3 (index2+1)
	where coordinates4= (posList2!!index2)
posFixer2p::[Coordinate] ->[[Cell]]->Int ->[(Prey,Coordinate)]
posFixer2p posList2 coordinates3 index2
	|null posList2 || null coordinates3 ||( index2>(length coordinates3)-1)= []
	|otherwise= [(function_p((coordinates3!!(snd coordinates4))!!(fst coordinates4)),coordinates4)]++ posFixer2p posList2 coordinates3 (index2+1)
	where coordinates4= (posList2!!index2)


--find the main list where the element resides
findPos :: Cell -> [[Cell]] -> Int -> Int -> Coordinate
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

findPosSpecial::Cell -> [[Cell]]->Int->Int->Coordinate
findPosSpecial H{} coords3 a b
	|null coords3 || (b> (length coords3)-1) = (-10,-10) -- add to list
	|H{} `elem` (coords3 !! b) = (findPosSpecial2 H{} (coords3!!b) 0 0 ,b) -- add to list
	|otherwise= findPosSpecial H{} coords3 a (b+1)  -- add to list
findPosSpecial P{} coords3 a b
	|null coords3 || (b> (length coords3)-1) = (-10,-10) -- add to list
	|P{} `elem` (coords3 !! b) =(findPosSpecial2 P{} (coords3!!b) 0 0 ,b) -- add to list
	|otherwise= findPosSpecial P{} coords3 a (b+1)  -- add to list


findPosSpecial2:: Cell -> [Cell] -> Int -> Int -> Int
findPosSpecial2 H{} coords4 a2 repeatNumber2
	|null coords4 || (a2> (length coords4) -1)= -10 
	|(coords4!!a2)==H{}   && repeatNumber2==0 = a2 
	|(coords4!!a2)==H{}  && repeatNumber2>0 = findPosSpecial2 H{} coords4 (a2+1) (repeatNumber2-1) 
	|otherwise= findPosSpecial2 H{} coords4 (a2+1) repeatNumber2
findPosSpecial2 P{} coords4 a2 repeatNumber2
	|null coords4 || (a2> (length coords4) -1)= -10 
	|(coords4!!a2)==P{}   && repeatNumber2==0 = a2 
	|(coords4!!a2)==P{}  && repeatNumber2>0 = findPosSpecial2 P{} coords4 (a2+1) (repeatNumber2-1) 
	|otherwise= findPosSpecial2 P{} coords4 (a2+1) repeatNumber2


--adds X and T's to theList
listAdder::[[Cell]]-> Int->Cell->[Coordinate]
listAdder coordsX rowNumber X
	|null coordsX = []
	|rowNumber>= length coordsX = []
	|X `elem` (coordsX !! rowNumber) = (listAdder2 rowNumber (coordsX!!rowNumber) 0 X) ++ (listAdder coordsX (rowNumber+1) X)
	|otherwise = listAdder coordsX (rowNumber+1) X
listAdder coordsX rowNumber T
	|null coordsX = []
	|rowNumber>= length coordsX = []
	|T `elem` (coordsX !! rowNumber) = (listAdder2 rowNumber (coordsX!!rowNumber) 0 T) ++ (listAdder coordsX (rowNumber+1) T)
	|otherwise = listAdder coordsX (rowNumber+1) T
listAdder coordsX rowNumber H{}
	|null coordsX = []
	|rowNumber>= length coordsX = []
	|H{} `elem` (coordsX !! rowNumber) = (listAdder2 rowNumber (coordsX!!rowNumber) 0 H{}) ++ (listAdder coordsX (rowNumber+1) H{})
	|otherwise = listAdder coordsX (rowNumber+1) H{}
listAdder coordsX rowNumber P{}
	|null coordsX = []
	|rowNumber>= length coordsX = []
	|P{} `elem` (coordsX !! rowNumber) = (listAdder2 rowNumber (coordsX!!rowNumber) 0 P{}) ++ (listAdder coordsX (rowNumber+1) P{})
	|otherwise = listAdder coordsX (rowNumber+1) P{}

	

--numBeforeX is for finding the next X with findPos2, rowNumber2 is for the y part of the tuple,
	-- miniCoordsX is the list where we search
listAdder2::Int->[Cell]->Int->Cell->[Coordinate]
listAdder2 rowNumber2 miniCoordsX numOfBeforeX X
	|null miniCoordsX =[(-10,-10)]
	|numOfBeforeX>=(length miniCoordsX) = []
	|otherwise = [(findPos2 X miniCoordsX 0 numOfBeforeX,rowNumber2)]++(listAdder2 rowNumber2 miniCoordsX (numOfBeforeX+1) X)
listAdder2 rowNumber2 miniCoordsX numOfBeforeX T
	|null miniCoordsX =[(-10,-10)]
	|numOfBeforeX>=(length miniCoordsX) = []
	|otherwise = [(findPos2 T miniCoordsX 0 numOfBeforeX,rowNumber2)]++(listAdder2 rowNumber2 miniCoordsX (numOfBeforeX+1) T)
listAdder2 rowNumber2 miniCoordsX numOfBeforeX H{}
	|null miniCoordsX =[(-10,-10)]
	|numOfBeforeX>=(length miniCoordsX) = []
	|otherwise = [(findPos2 H{} miniCoordsX 0 numOfBeforeX,rowNumber2)]++(listAdder2 rowNumber2 miniCoordsX (numOfBeforeX+1) H{})
listAdder2 rowNumber2 miniCoordsX numOfBeforeX P{}
	|null miniCoordsX =[(-10,-10)]
	|numOfBeforeX>=(length miniCoordsX) = []
	|otherwise = [(findPos2 P{} miniCoordsX 0 numOfBeforeX,rowNumber2)]++(listAdder2 rowNumber2 miniCoordsX (numOfBeforeX+1) P{})


directionAdder::[[Cell]]->Cell->[Coordinate]->Int->[[Direction]]
directionAdder coords5 H{} hlocations index
	|null coords5 || null hlocations || index>(length hlocations)-1=[[]]
	|otherwise= [fugazi_actions (function_h(elementPosition))]++directionAdder coords5 H{} hlocations (index+1)
	where listtuple=(hlocations!!index)
	      elementPosition=(coords5!!(snd (listtuple)) )!!(fst listtuple)
directionAdder coords5 P{} hlocations index
	|null coords5 || null hlocations || index>(length hlocations)-1=[[]]
	|otherwise= [fugazi_actions2 (function_p(elementPosition))]++directionAdder coords5 P{} hlocations (index+1)
	where listtuple=hlocations!!index
	      elementPosition=(coords5!!(snd (listtuple)) )!!(fst listtuple)



fugazi_actions (Hunter _ _ _ _ act) = act
fugazi_actions2 (Prey _ _ act) = act
function_h (H x)  = x 
function_p (P x)  = x





{-
	make simulate
	||||||
	

-}