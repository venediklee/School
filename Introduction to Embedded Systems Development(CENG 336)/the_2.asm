#include "p18f8722.inc"

    CONFIG OSC=HSPLL, FCMEN=OFF, IESO=OFF,PWRT=OFF,BOREN=OFF, WDT=OFF, MCLRE=ON, LPT1OSC=OFF, LVP=OFF, XINST=OFF, DEBUG=OFF
;*******************************************************************************
; Variables & Constants
;*******************************************************************************
; variables 
    ; health
    ; level
    ; timer1 starting value
    ; number of balls spawned in that level
    ; 15bits to determine which balls are active(5-10-15 are used for each level)
    ; 15 times 5bits for determining where the balls are
    UDATA_ACS
pressed res 1 ; pressed[0] := RG0, pressed[2] := RG2, pressed[3] := RG3    
health res 1; 7 segment display of health is at portH.0 -> b'00000001' = 1
level res 1; 7 segment display of level is at portH.1 -> b'00000010' = 2
timer_counter res 1
timer_state res 1 ; this is set when the required time has passed for each level
numberOfSpawnedBalls res 1
activeBallsSet1 res 1 ; 5 balls. only use rightmost 5 bits
activeBallsSet2 res 1 ; 5 balls. only use rightmost 5 bits after level-1
activeBallsSet3 res 1 ; 5 balls. only use rightmost 5 bits after level-2
ball1Position res 1 ; 0 indicates top left, 23 indicates bottom right. Add 4 per update.
ball2Position res 1
ball3Position res 1 
ball4Position res 1
ball5Position res 1 
ball6Position res 1
ball7Position res 1 
ball8Position res 1
ball9Position res 1 
ball10Position res 1
ball11Position res 1 
ball12Position res 1
ball13Position res 1 
ball14Position res 1
ball15Position res 1
barPosition res 1 ; Leftmost bar position. Between 20-22(inclusive) for easier comparison. 20 means bar is at 20 and 21'st points. 22 means bar is at 22nd and 23th points.
 
;*******************************************************************************
; Reset Vector
;*******************************************************************************

RES_VECT  CODE    0x0000            ; processor reset vector
    GOTO    main                   ; go to beginning of program

;Interrupt Vector    

org     0x08
goto    isr             ;go to interrupt service routine
    
;*******************************************************************************
; MAIN PROGRAM
;*******************************************************************************

isr:
    ;btsfs INTCON, 2 ; TMR0IF is bit 2
    retfie ;some other interrupt, should not happen return
    decf	timer_counter, f              ;Timer interrupt handler part begins here by decrementing count variable
    btfss	STATUS, Z               ;Is the result Zero?
    goto	timer_interrupt_exit    ;No, then exit from interrupt service routine
    clrf	timer_counter                 ;Yes, then clear count variable
    comf	timer_state, f                ;Complement our state variable
    ;TO-DO: handle different cases for level here and set timer_counter to some initial value

timer_interrupt_exit:
    bcf		INTCON, 2		    ;Clear TMROIF
    movlw	d'61'               ;256-61=195; 195*256*100 = 4992000 instruction cycle;
    movwf	TMR0
    ;call	restore_registers   ;Restore STATUS and PCLATH registers to their state before interrupt occurs
    retfie
    
;NOTE: the +-100 ms might be the same for all balls at each update or it might be unique to each ball at each update.

; <X> indicates X is a label/state etc.
; "X" indicates X is a variable

; initialize 
; -> set the bar at RA5 & RB5
; -> set level to 1 at D3 of 7segment display
; -> set health to 5 at D0 of 7segment display
; set "ball update period" to its new value
; -> goto <start>
initialize
    ;setup timer
    clrf TMR0; TMR0 = 0
    clrf INTCON; Interrupts disabled for now
    movlw b'11010111'; enable timer, 8-bit operation, ; falling edge, select prescaler ; with 1:256, internal source
    movwf T0CON; T0CON <-W
    movlw d'61'; 10MHZ clock -> 10^7 cycles per second -> 10^-4 ms per cycle; 
    movwf TMR0L; counter can count x*256 cycles -> x*256*10^-4 ms -> x=195 for 4,992 ms -> 256-195 = 61 = '0x3d'
    movlw d'100'; 100*4,992= 499,2 ms is passed to the counter to count ~500ms for level1
    movwf timer_counter
    ;setup ports, inputs-outputs etc.
    movlw 0x0F
    movwf ADCON1 ; set A/D conversion
    movlw b'00001101' ; RG0-RG2-RG3 are input 
    movwf TRISG
    clrf LATG ; clear port G content just in case TODO can we clear without reading?
    clrf TRISA; RA0-RA5, RB0-RB5, RC0-RC5, RD0-RD5 are outputs
    clrf TRISB
    clrf TRISC
    clrf TRISD
    clrf LATA ; clear output port content just in case TODO can we clear without reading?
    clrf LATB
    clrf LATC
    clrf LATD
    clrf TRISH ; porth and portj are 7segment display(outputs)
    clrf TRISJ
    clrf LATH
    clrf LATJ
    ;set variables
    movlw 5
    movwf health
    movlw 1
    movwf level
    movlw 1
    movwf TRISH ; enable first 7segment display for setting health
    movlw b'01101101' ;5 for 7segment display
    movwf LATJ ; TODO we may need to movwf to TRISJ instead
    nop ;it says wait a while on the hw pdf
    clrf TRISH
    clrf LATJ
    movlw 2
    movwf TRISH ;enable second 7segment display for setting level
    movlw b'00000110' ; 1 for 7segment display
    movwf LATJ ; TODO we may need to movwf to TRISJ instead
    nop ;it says wait a while on the hw pdf
    clrf TRISH
    clrf LATJ
    ;Enable interrupts
    movlw   b'11100000' ;Enable Global, peripheral, Timer0 by setting GIE, PEIE, TMR0IE bits to 1
    movwf   INTCON
    bsf     T0CON, 7    ;Enable Timer0 by setting TMR0ON to 1
    
    clrf numberOfSpawnedBalls
    clrf activeBallsSet1
    clrf activeBallsSet2
    clrf activeBallsSet3
    movlw 20
    movwf barPosition
    ;mowlw b'00100000'
    movwf LATA ; light the bar
    movwf LATB ; light the bar
    return


; -> save timer1 value(16 bit)
; wait for RGO, if it is pressed and released goto loop
wait_rg0_press:			;TODO save timer1 value
    btfsc pressed, 0
    goto wait_rg0_release
    btfss PORTG, 0
    goto wait_rg0_press
    bsf pressed, 0
    
wait_rg0_release:
    btfsc PORTG, 0
    goto wait_rg0_release
    bcf pressed, 0
    goto loop
    
   
    
; move the bar
; -> if RG2 is pressed & bar is not at RE5-RF5 move right. Reset RG2 to 0 and goto <move the active balls>.
; -> if RG3 is pressed & bar is not at RA5-RB5 move left. Reset RG3 to 0 and goto <move the active balls>.
; ->goto <move the active balls>
moveTheBar
    btfsc PORTG,2 ; if RG2 is NOT pressed don't execute move right
    goto moveRight
    btfsc PORTG,3 ; if RG3 is NOT pressed don't execute move left
    goto moveLeft
    return
    
    moveRight:
	movlw 22
	cpfslt barPosition ; skip if we are already on the rightmost position
	return ;(barPosition=22)
	incf barPosition
	goto lightTheBar
	
    moveLeft:
	movlw 20
	cpfsgt barPosition ;skip if we are already on the leftmost position
	return ;(barPosition=20)
	decf barPosition
	goto lightTheBar
	
    lightTheBar:
	;TODO light the bar
	; only the 5th light of A-F will be on (don't forget to close the previous light positions)
	
	;mowlw b'00000000' ; reset led not to keep previous data (we can change the design)
	movwf	LATA
	movwf	LATB
	movwf	LATC
	movwf	LATD    
    
	case20:		; case for bar=20
	    movlw 20
	    cpfseq barPosition
	    goto case21
	    movlw b'00100000'
	    movwf LATA
	    movwf LATB
	    return
	case21:		; case for bar=21
	    movlw 21
	    cpfseq barPosition
	    goto case22
	    movlw b'00100000'
	    movwf LATB
	    movwf LATC
	    return

	case22:		; case for bar=22
	    movlw 22
	    cpfseq barPosition
	    goto case20
	    movlw b'00100000'
	    movwf LATC
	    movwf LATD
	    return
	
	
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


idle:  ; restart part is here too
    ; TODO set "15 bit active balls" to 0
    ; TODO set "ball update period" to its new value
    clrf pressed
    clrf numberOfSpawnedBalls 
    movlw 1
    movwf level
    movlw 5
    movwf health
    clrf LATG
    goto wait_rg0_press
    
main
    call initialize
    goto idle
    loop:
	call moveTheBar
	
    goto loop
    END

;;;;;;;;;;;; Register handling for proper operation of main program ;;;;;;;;;;;;
save_registers:
    movwf 	w_temp          ;Copy W to TEMP register
    swapf 	STATUS, w       ;Swap status to be saved into W
    clrf 	STATUS          ;bank 0, regardless of current bank, Clears IRP,RP1,RP0
    movwf 	status_temp     ;Save status to bank zero STATUS_TEMP register
    movf 	PCLATH, w       ;Only required if using pages 1, 2 and/or 3
    movwf 	pclath_temp     ;Save PCLATH into W
    clrf 	PCLATH          ;Page zero, regardless of current page
	return

restore_registers:
    movf 	pclath_temp, w  ;Restore PCLATH
    movwf 	PCLATH          ;Move W into PCLATH
    swapf 	status_temp, w  ;Swap STATUS_TEMP register into W
    movwf 	STATUS          ;Move W into STATUS register
    swapf 	w_temp, f       ;Swap W_TEMP
    swapf 	w_temp, w       ;Swap W_TEMP into W
    return
