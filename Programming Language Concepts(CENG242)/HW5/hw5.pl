:-module(hw5, [configuration/3]).
:-[hardware].

% empty components means empty placement list.
configuration([],_,PlacementList):- false.

% no constraints and one element:
%%%% PlacementList is the element with all the sections.
configuration([A],[],PlacementList):- hardware:sections(SectionList),
                                      sectionSplitter(SectionList,Sect),
                                      PlacementList=[put(A,Sect)].




% one(outer edge) constraint and one element
configuration([A],[outer_edge(A)],PlacementList):-
                                                 hardware:sections(SectionList),
                                                 length(SectionList,Slength),
                                                 sectionSplitter(SectionList,Sect),
                                                 (Slength<5->configuration([A],[],PlacementList);
                                                 (outerSect(Sect)->PlacementList=[put(A,Sect)];false)).
% one(close to) constraint and 2 elements
configuration([A,B],[close_to(A,B)],PlacementList):-
                                                    hardware:sections(SectionList),
                                                    sectionSplitter(SectionList,Sect),
                                                    sectionSplitter(SectionList,Sect2),
                                                    Sect\=Sect2,
                                                    ((hardware:adjacent(Sect,Sect2);hardware:adjacent(Sect2,Sect))->
                                                    PlacementList=[put(A,Sect),put(B,Sect2)];false).


configuration([A,B],[close_to(B,A)],PlacementList):-configuration([A,B],[close_to(A,B)],PlacementList).

% if there are more elements then sections
configuration(Elements,_,_):-
                                          hardware:sections(SectionList),
                                          length(SectionList,Slength),
                                          length(Elements,Elength),
                                          (Elength>Slength, false).


% if there are more outer edge constraints then outer edge sections
% (len constraints=>len elements)
configuration(Elements,Constraints,_):-length(Elements,Elength),length(Constraints,Clength),(Clength>=Elength,false).

% two or more elements no constraints
configuration(Elements,[],PlacementList):-
                                          !,hardware:sections(SectionList),
                                           configuration2(Elements,[],PlacementList,SectionList).


configuration(Elements,A,PlacementList):-configuration(Elements,[],PlacementList),!.




configuration2([],_,_,_):-!.
configuration2([E],[],PlacementList,NewSections):-
                                                  !,newSectSelector(NewSections,Sect),
                                                  PlacementList=[put(E,Sect)].
configuration2([E|ERest],[],PlacementList,NewSections):-
                                                          newSectSelector(NewSections,Sect),
                                                          select(Sect,NewSections,NewSections2),
                                                          PlacementList3=[put(E,Sect)],
                                                          configuration2(ERest,[],PlacementList2,NewSections2),
                                                          append(PlacementList3,PlacementList2,PlacementList).


newSectSelector([S],Sect):-Sect=S,!.
newSectSelector([S|SR],Sect):-Sect=S;sectionSplitter(SR,Sect).




% checks if the Sect is outer
outerSect(Sect):-
                 hardware:sections(SectionList),
                 sectionSplitter(SectionList,Sect2),
                 sectionSplitter(SectionList,Sect3),
                 sectionSplitter(SectionList,Sect4),
                 sectionSplitter(SectionList,Sect5),
                 Sect\=Sect2, Sect\=Sect3,Sect\=Sect4,Sect\=Sect5,
                 Sect2\=Sect3, Sect2\=Sect4,Sect2\=Sect5,
                 Sect3\=Sect4, Sect3\=Sect5,
                 Sect4\=Sect5,
                 (doubleAdjacent(Sect,Sect2),doubleAdjacent(Sect,Sect3),doubleAdjacent(Sect,Sect4),
                  doubleAdjacent(Sect,Sect5) ->false;true).


doubleAdjacent(Sect,Sect2):-hardware:adjacent(Sect,Sect2);hardware:adjacent(Sect2,Sect).


sectionSplitter([],_):-!.
sectionSplitter([A],Sect):- Sect = A, !.
sectionSplitter([A|Rest],Sect) :- Sect=A;sectionSplitter(Rest,Sect).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
/*
% else
configuration([E|ERest],Constraints,PlacementList):-
                                                 (member(outer_edge(E),Constraints)->configuration([E],[outer_edge(E)],PlacementList2);
                                                 (member(close_to(E,E2),Constraints)->configuration([E,E2],[close_to(E,E2)],PlacementList2);
                                                        configuration([E],[],PlacementList2))),
                                                 configuration(ERest,Constraints,PlacementList3),
                                                 append(PlacementList2,PlacementList3,PlacementList).
*/




/*


% if there are more outer edge constraints then outer edge sections
configuration(_,Constraints,_):-
                                       hardware:sections(SectionList),
                                       outerSectCounter2(SectionList,Num),
                                       outerConstrCounter2(Constraints,Num2),
                                       (Num2>Num,false).





% if there are more close to constraints then possible close to sections


% counts outer sections
%outerSectCounter(A,outerSectCounter2(A,0)).
outerSectCounter2([],Num):-Num.
outerSectCounter2([A],Num):- (outerSect(A)->outerSectCounter2([],Num+1);outerSectCounter2([],Num)).
outerSectCounter2([A|ARest],Num):-(outerSect(A)->outerSectCounter2([ARest],Num+1);outerSectCounter2([ARest],Num)).
% counts outer constraints
%outerConstrCounter(A, outerConstrCounter2(A,0)).
outerConstrCounter2([],Num):-Num,!.
outerConstrCounter2([outer_edge(_)],Num):-outerConstrCounter2([],Num+1).
outerConstrCounter2([outer_edge(_)|ARest],Num):-outerConstrCounter2([ARest],Num+1).
outerConstrCounter2([close_to(_)],Num):-outerConstrCounter2([],Num).
outerConstrCounter2([close_to(_)|ARest],Num):-outerConstrCounter2([ARest],Num).


*/








/*



% one(outer edge) constraint and one element:
%%%% You should find the outer edges from hardware.pl
%%%%% If a sections name is given twice or more in Adjacents-> IT IS NOT OUTER EDGE.
configuration([A],[OuterEdgeConstraintOfA],PlacementList):- hardware:sections(SectionList),
                                                getAdjacents(Adjacents),
                                                outerSectionDeletor(Sect,Adjacents,SectionList),
                                                PlacementList=[put(A,Sect)].
getAdjacents(Adjacents):-hardware:adjacent(Adj1,Adj2),append([Adj1],[Adj2],Adjacents).

% if there is no Adjacents or one Adjacents->do sectionSplitter
outerSectionDeletor(Sect,[],MySectionList):- sectionSplitter(MySectionList,Sect),!.
outerSectionDeletor(Sect,[A],MySectionList):- sectionSplitter(MySectionList,Sect),!.

% if there is 2n(+1) Adjacents->check if a section name repeats itself,
                                      % if so delete that section from MySectionList
                                      % else continue searching
outerSectionDeletor(Sect,[Adj1|AdjRest],MySectionList):- member(Adj1,AdjRest),delete(MySectionList,Adj1,NewMySectionList),
                                                      delete(AdjRest,Adj1,NewAdjRest),   % <- for faster computation
                                                      outerSectionDeletor(Sect,NewAdjRest,NewMySectionList);
                                                      outerSectionDeletor(Sect,AdjRest,MySectionList).











*/


/*


% SectionList, a Section that must be outer edge, Adjacent section list
%%% 'select' the element from Adjacents and check if it is a member of the newAdjacents list
                                                      % if it is delete it from MySectionList
          % no adjacents-> do sectionSplitter
outerSectionDeletor(SectionList,Sect,[],MySectionList):-sectionSplitter(MySectionList,Sect).

          % an adjacent

outerSectionDeletor(SectionList,Sect,Adjacents,MySectionList):-
*/
/*
?- configuration([fan], [outer_edge(fan)], PlacementList).
PlacementList = [put(fan, sA)] ;
PlacementList = [put(fan, sB)] ;
PlacementList = [put(fan, sD)] ;
PlacementList = [put(fan, sE)].
*/
/*
*      :- module(hardware, [sections/1, adjacent/2]).
*
*      sections([sA, sB, sC, sD, sE]).
*      adjacent(sA,sC).
*      adjacent(sB,sC).
*      adjacent(sC,sD).
*      adjacent(sC,sE).
*/

