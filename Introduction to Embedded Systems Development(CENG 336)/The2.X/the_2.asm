#include "p18f8722.inc"

    CONFIG OSC=HSPLL, FCMEN=OFF, IESO=OFF,PWRT=OFF,BOREN=OFF, WDT=OFF, MCLRE=ON, LPT1OSC=OFF, LVP=OFF, XINST=OFF, DEBUG=OFF
;*******************************************************************************
; Variables & Constants
;*******************************************************************************
    UDATA_ACS
;t1	res 1	; used in delay
 
;*******************************************************************************
; Reset Vector
;*******************************************************************************

RES_VECT  CODE    0x0000            ; processor reset vector
    GOTO    main                   ; go to beginning of program

;*******************************************************************************
; MAIN PROGRAM
;*******************************************************************************
	
    
    
;NOTE: the +-100 ms might be the same for all balls at each update or it might be unique to each ball at each update.

; <X> indicates X is a label/state etc.
; "X" indicates X is a variable
   
    
; variables 
; health
; level
; timer1 starting value
; number of balls spawned in that level
; ball update period(500-400-350 ms for each level). don't forget the +-100ms for moving the balls. 
; 15bits to determine which balls are active(5-10-15 are used for each level)
; 15 times 6bits for determining where the balls are

; initialize 
; -> set the bar at RA5 & RB5
; -> set level to 1 at D3 of 7segment display
; -> set health to 5 at D0 of 7segment display
; set "ball update period" to its new value
; -> goto <start>

; start
; -> if RG0 is never pressed goto start
; -> if pressed before, goto <move the bar>
; -> save timer1 value(16 bit)
; -> goto <move the bar>

; move the bar
; -> if RG2 is pressed & bar is not at RE5-RF5 move right. Reset RG2 to 0 and goto <move the active balls>.
; -> if RG3 is pressed & bar is not at RA5-RB5 move left. Reset RG3 to 0 and goto <move the active balls>.
; ->goto <move the active balls>

; move the active balls
; -> if "ball update period" +-100ms passed
;	-> for each active ball(can find with "15bit active balls")
;	    -> add 6 to the corresponding "6bit ball location"
; -> goto <check active balls>

; check active balls
; -> for each active ball(can find with "15bit active balls")
;	-> if "6bit ball location" >= 36, i.e ball hasn't been caught
;	    -> decrement "health" and update 7segment display
;	    -> if "health" == 0, goto <restart>	----------- <- may also goto <lose> if we want to wait for sometime
;	    -> deactivate the ball by updating "15bit active balls"
;	-> else if "6bit ball location" >=30 & ball is on the bar, i.e ball has been caught
;	    -> deactivate the ball by updating "15bit active balls"
; -> if "15 bit active balls" == 0, goto <next level>
; -> goto <create the balls>

; create the balls
; -> if we need to create more balls and  "ball update period" ms passed
;	-> find a non active ball index from "15bit active balls". (can use "number of spawned balls")
;	    -> Set that index to 1.
;	    -> Set the "6bit ball location" corresponding to that index to [0,5] based on "timer1 starting value" & timer0
;	    -> increase "number of spawned balls" by 1
; -> goto <move the bar>

; next level
; if "level" == 3, goto <restart>
; set "number of spawned balls" to 0
; increment "level" by 1 and update 7segment display
; set "ball update period" to its new value
; goto <move the bar>

; restart 
; set "number of spawned balls" to 0
; set "15 bit active balls" to 0
; set "level" to 1 and update 7segment display
; set "health" to 5 and update 7segment display
; set "ball update period" to its new value
; set RG0 to 0
; goto <start>

; lose(OPTIONAL?)
; wait for some
; goto <restart>
    
main
    call initialize    
    END
