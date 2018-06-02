import Data.List -- YOU MAY USE THIS MODULE FOR SORTING THE AGENTS

data Level = Newbie | Intermediate | Expert deriving (Enum, Eq, Ord, Show, Read)
data Hunter = Hunter {hID::Int, hlevel::Level, hEnergy::Int, hNumberOfCatches::Int, hActions::[Direction]} deriving (Eq, Show, Read)
data Prey = Prey {pID::Int, pEnergy::Int, pActions::[Direction]} deriving (Eq, Show, Read)
data Cell = O | X | H Hunter | P Prey | T deriving (Eq, Show, Read)
data Direction = N | S | E | W deriving (Eq, Show, Read)
type Coordinate = (Int, Int)

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

let func a=5*b 
	where b=2



		--simulate [ [ P ( Prey 1 70 [ S , E , W , S , N , E ] ) , O , X , P ( Prey 2 100 [ N , N , W , S , E , E ] ) ] ,[ T , O , H ( Hunter 1 Expert 90 2 [ E , S , N , E , S , W ] ) , O ] , [ H ( Hunter 2 Newbie 70 0 [ E , N ,S , W , N , N ] ) , O , X , O ] ]

		--sort [ Hunter 1 Expert 100 10 [ N , W ] , Hunter 2 Expert 80 5 [ E , S ] , Hunter 3Newbie 80 1 [ N , N ] , Hunter 4 Expert 100 10 [ S , W ] ]