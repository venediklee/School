using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityStandardAssets.CrossPlatformInput;
using UnityEngine.UI;


//using cyclone;

public class PlayerController : MonoBehaviour
{
    //Handles player movement
    //players hands will be held still on the handles of the wall until the hand gets tired or the forces are too extreme*
    //players feet will be held on top of the handles of the wall indefinetly
    //>user needs to press and hold mouse buttons to throw** players hand above/below the current position, direction is determined with camera angle
    //>user needs to move players feet to on top of another handle or use the walls friction to push the player up/down
    //>>movement is done with WASD when not on walls, with space&ctrl when on walls

    //*extremeness of the forces will be determined like a spring; if the body is only held with one or two arms, the whole weight will be on those arms
    //>or if the hands and feet are too close the player will fall
    //**holding are done in gang beasts/human fall flat type of movement; throwing is done when mouse button(s) is held



    //////int h,v,d;//used for movement axis of feet, h=horizontal(sideways)->a&d,  v=vertical(upwards)->s&w,  d=depth(inwards)->q&e <<<<-1&1 respectively 
    //////if (Input.GetKey("a")) h = -1;
    //////else if (Input.GetKey("d")) h = 1;
    //////else h = 0;

    //////if (Input.GetKey("s")) v = -1;
    //////else if (Input.GetKey("w")) v = 1;
    //////else v = 0;

    //////if (Input.GetKey("q")) d = -1;
    //////else if (Input.GetKey("e")) d = 1;
    //////else d = 0;

    float playerMovementSpeedMutliplier = 3f;

    [SerializeField] Transform playerCam, playerCamSetup;
    [SerializeField] Transform player,rightHand, leftHand, rightFoot, leftFoot;
    [SerializeField] LayerMask wallsAndHandles, handleAttachPositions;

    /// <summary>
    /// the point where the camera looks(calculated with raycast), used for determining movement direction of hands 
    /// </summary>
    Vector3 lookPoint;
    /// <summary>
    /// movement direction of hands
    /// </summary>
    cyclone.MyVector3 rightHandMovementDirection, leftHandMovementDirection;
    RaycastHit hit;

    //TODO DECISION power increase
    [Tooltip("stores the amount of power each hand has, determines total 'hand' throwing distance, pressing R resets to 0")]
    [SerializeField] [Range(0, 100)] float rightHandPower = 0, leftHandPower = 0;
    /// <summary>
    /// wheter or not the hand is holding the handles etc.
    /// </summary>
    bool isRightHandAttached = true, isLeftHandAttached = true;
    /// <summary>
    /// wheter or not the hand is moving, doesn't change when walking
    /// </summary>
    bool isRightHandMoving = false, isLeftHandMoving = false;

    /// <summary>
    /// particles that are used for simulating physics of hands/feet
    /// </summary>
    cyclone.Particle LHParticle, RHParticle, LFParticle, RFParticle;

    /// <summary>
    /// the offset of camera based on initial average points of hands & feet
    /// </summary>
    //cyclone.MyVector3 cameraOffset = new cyclone.MyVector3();

    /// <summary>
    /// the offset of player(parent) based on initial average points of hands & feet
    /// </summary> 
    cyclone.MyVector3 playerOffset=new cyclone.MyVector3();

    /// <summary>
    /// players stamine, decreases per movement, increases overtime if both hands and feet are holding somewhere
    /// </summary>
    [SerializeField] float stamina = 100f;

    /// <summary>
    /// used for power GFX
    /// </summary>
    [SerializeField] RectTransform leftPowerGFX, rightPowerGFX;
    /// <summary>
    /// used for stamina GFX
    /// </summary>
    [SerializeField] RectTransform staminaGFX;

    private void Start()
    {
        InitHandFeetParticles();//initialize hand/feet particles

        //cameraOffset = (LHParticle.GetPosition() + RHParticle.GetPosition() + LFParticle.GetPosition() + RFParticle.GetPosition()) * 0.25f;
        cyclone.MyVector3 rhp = new cyclone.MyVector3();rhp.x= RHParticle.position.x; rhp.y = RHParticle.position.y; rhp.z = RHParticle.position.z;
        playerOffset = cyclone.MyVector3.ConvertToMyVector3(player.position) - (RHParticle.position + LHParticle.position + 
            RFParticle.position + LFParticle.position) * 0.25 ;
        RHParticle.position = rhp;
    }
    

    //TODO feet climbing
    //TODO add falling mechanisms-> stamina(decreases per move+over time & replenishes when shaking hands etc.), odd positions decrease stamina faster
    //TODO add left/right hand power & stamina GFX
    //TODO LATER convert add force of hands to impulse?
    //TODO (add body GFX?)
    private void Update()
    {

        staminaGFX.sizeDelta = new Vector2(stamina, staminaGFX.sizeDelta.y);
        leftPowerGFX.sizeDelta= new Vector2(leftPowerGFX.sizeDelta.x, leftHandPower);
        rightPowerGFX.sizeDelta= new Vector2(rightPowerGFX.sizeDelta.x, rightHandPower);

        //fall down if stamina reached less than 0 or both hands are moving 
        if (stamina < 0 || (isLeftHandAttached==false && isRightHandAttached==false)) 
        {
            Debug.Log("falling down, stamina="+stamina);
            FallDown();
        }
        

        if (isRightHandAttached == false && isLeftHandAttached == false &&
            isLeftHandMoving == false && isRightHandMoving == false)  //player can walk
        {
            float horizontal, vertical;
            horizontal = CrossPlatformInputManager.GetAxis("Horizontal")*Time.deltaTime*playerMovementSpeedMutliplier;
            vertical = CrossPlatformInputManager.GetAxis("Vertical")*Time.deltaTime*playerMovementSpeedMutliplier;
            //update player's position
            player.transform.position += new Vector3(vertical, 0, -horizontal);
            //also need to update hands/feets particle positions
            LHParticle.position += new cyclone.MyVector3(vertical, 0, -horizontal);
            RHParticle.position += new cyclone.MyVector3(vertical, 0, -horizontal);
            LFParticle.position += new cyclone.MyVector3(vertical, 0, -horizontal);
            RFParticle.position += new cyclone.MyVector3(vertical, 0, -horizontal);
        }
        else
        {
            //TODO change camera's position calculations 
            //update player's(therefore camera's) position based on positions of hands and feet
            //////broken-> lhparticle.position changes
            //////player.position = cyclone.MyVector3.ConvertToVector3(
            //////    (LHParticle.GetPosition() + RHParticle.GetPosition() + LFParticle.GetPosition() + RFParticle.GetPosition()) * 0.25);
            //////Debug.Log("changing camera position->"+playerCam.position);

            //move the cam setup
            float horizontal, vertical;
            horizontal = CrossPlatformInputManager.GetAxis("Horizontal") * Time.deltaTime * playerMovementSpeedMutliplier;
            vertical = CrossPlatformInputManager.GetAxis("Vertical") * Time.deltaTime * playerMovementSpeedMutliplier;
            playerCamSetup.position+= new Vector3(vertical, 0, -horizontal);
        }

        if (Input.GetKey(KeyCode.Mouse0))
        {
            //TODO DECISION speed increase
            //add power to left hand(max power in 3 seconds,upgrades decrease this) , apply force-in the direction of the camera- when lifted
            if(!isLeftHandMoving) leftHandPower += Time.deltaTime * 100/3;
            if (leftHandPower > 100) leftHandPower = 100;
        }
        if (Input.GetKey(KeyCode.Mouse1))
        {
            //TODO DECISION  speed increase
            //add power to right hand, apply force-in the direction of the camera- when lifted
            if(!isRightHandMoving) rightHandPower += Time.deltaTime * 100 / 3;
            if (rightHandPower > 100) rightHandPower = 100;
        }
        if (Input.GetKeyUp(KeyCode.Mouse0))
        {
            //apply accumulated force to left hand in the direction of the camera
            Physics.Raycast(playerCam.position, playerCam.forward, out hit, 10.0f, wallsAndHandles);
            Debug.DrawRay(playerCam.position, playerCam.forward*10, Color.yellow,5);
            if (hit.collider!=null)//move only if there is a wall or handle
            {
                isLeftHandAttached = false;
                isLeftHandMoving = true;
                StartCoroutine(ChangeHandMovementStatus(false, false, 0.35f));

                leftHandMovementDirection = cyclone.MyVector3.ConvertToMyVector3(hit.point - leftHand.position);
                //add force
                LHParticle.AddForce(leftHandMovementDirection.Normalized()*leftHandPower);
                StartCoroutine(LHParticle.DecreaseVelocityAfterTime(0.35f));
                StartCoroutine(LHParticle.DecreaseAccelerationAfterTime(0.35f));
                StartCoroutine(ActivateHandleHolding(false, 0.35f));

                //decrease stamina
                stamina -= 20f * leftHandPower / 100;
            }
            leftHandPower = 0;
        }
        if (Input.GetKeyUp(KeyCode.Mouse1))
        {
            //apply accumulated force to right hand in the direction of the camera
            Physics.Raycast(playerCam.position, playerCam.forward, out hit,10.0f, wallsAndHandles);
            Debug.DrawRay(playerCam.position, playerCam.forward*10, Color.yellow,5);
            if(hit.collider!=null)//move only if there is a wall or handle
            {
                isRightHandAttached = false;
                isRightHandMoving = true;
                StartCoroutine(ChangeHandMovementStatus(true, false, 0.35f));

                rightHandMovementDirection = cyclone.MyVector3.ConvertToMyVector3(hit.point - rightHand.position);
                //add force
                RHParticle.AddForce(rightHandMovementDirection.Normalized()*rightHandPower);
                StartCoroutine(RHParticle.DecreaseVelocityAfterTime(0.35f));
                StartCoroutine(RHParticle.DecreaseAccelerationAfterTime(0.35f));
                StartCoroutine(ActivateHandleHolding(true, 0.35f));
                
                //decrease stamina
                stamina -= 20f * rightHandPower / 100;
            }
            rightHandPower = 0;
        }

        if(Input.GetKeyDown(KeyCode.R))
        {
            //reset hand powers
            rightHandPower = 0;
            leftHandPower = 0;
        }

        if (Input.GetKeyDown(KeyCode.Space))
        {
            //if hands are holding anything-> pull yourself up, then put feet in 'safest' position(s)
            if(isLeftHandAttached==true || isRightHandAttached==true)
            {
                Collider[] bestPossiblePositions = Physics.OverlapSphere((leftHand.position+rightHand.position)/2, 5f, handleAttachPositions);
                if(bestPossiblePositions.Length==0)
                {
                    //TODO climbing on the wall things
                }
                else
                {
                    float minDistanceSqred = (bestPossiblePositions[0].transform.position - (rightFoot.position + leftFoot.position) / 2).sqrMagnitude;
                    float nextMinDistanceSqred = 0;
                    int closestPositionIndex = 0;
                    for (int i = 1; i < bestPossiblePositions.Length; i++)//don't need to re-calculate first distance sqred
                    {
                        nextMinDistanceSqred = (bestPossiblePositions[i].transform.position - (rightFoot.position + leftFoot.position) / 2).sqrMagnitude;
                        if(nextMinDistanceSqred<minDistanceSqred)
                        {
                            minDistanceSqred = nextMinDistanceSqred;
                            closestPositionIndex = i;
                        }
                    }

                    //decrease stamina, max 30, min 0, distance(0,5^2)
                    stamina -= 30f*0.04f*  (float) (cyclone.MyVector3.ConvertToMyVector3(bestPossiblePositions[closestPositionIndex].transform.position) 
                        - (RFParticle.GetPosition() + LFParticle.GetPosition()) / 2).SquareMagnitude();
                    

                    //TODO LATER put feet in different positions
                    //put the feet on the closest possible position
                    RFParticle.SetPosition(bestPossiblePositions[closestPositionIndex].transform.position);
                    LFParticle.SetPosition(bestPossiblePositions[closestPositionIndex].transform.position);

                    //move the player so the camera moves as well
                    cyclone.MyVector3 rhp = new cyclone.MyVector3(); rhp.x = RHParticle.position.x; rhp.y = RHParticle.position.y; rhp.z = RHParticle.position.z;
                    cyclone.MyVector3 lhp = new cyclone.MyVector3(); lhp.x = LHParticle.position.x; lhp.y = LHParticle.position.y; lhp.z = LHParticle.position.z;
                    player.position = cyclone.MyVector3.ConvertToVector3((RHParticle.position + LHParticle.position + 
                        RFParticle.position + LFParticle.position) * 0.25 + playerOffset) ;
                    RHParticle.position = rhp;
                    LHParticle.position = lhp;
                }
            }
        }

        if (Input.GetKeyDown(KeyCode.LeftControl))
        {
            //(hinge movement)
            //if hands are holding anything && we pulled ourself up-> pull yourself down
            //if hands are holding anything && we didnt pull ourself up-> go down a step
        }


        //increase stamina if both of our hands (and feet are attached - which is always true)
        if (isRightHandAttached == true && isLeftHandAttached == true)
        {
            Debug.Log("increasing stamina");
            stamina += Time.deltaTime * 100f / 10f;//recharges 100 percent in 10 seconds
            if (stamina > 100) stamina = 100;
        }
    }

    

    private void FixedUpdate()
    {
        //update particles
        ParticleUpdate();
        
    }

    /// <summary>
    /// initializes hand and feet particles  
    /// </summary>
    private void InitHandFeetParticles()
    {
        LHParticle = new cyclone.Particle();
        LHParticle.SetMass(2.0f); // 2.0kg
        LHParticle.SetVelocity(0.0f, 0.0f, 0.0f); // 0m/s
        LHParticle.SetAcceleration(0.0f, 0.0f, 0.0f);
        LHParticle.SetDamping(0.99f);
        LHParticle.SetPosition(leftHand.position);

        RHParticle = new cyclone.Particle();
        RHParticle.SetMass(2.0f); // 2.0kg
        RHParticle.SetVelocity(0.0f, 0.0f, 0.0f); // 0m/s
        RHParticle.SetAcceleration(0.0f, 0.0f, 0.0f);
        RHParticle.SetDamping(0.99f);
        RHParticle.SetPosition(rightHand.position);

        LFParticle = new cyclone.Particle();
        LFParticle.SetMass(2.0f); // 2.0kg
        LFParticle.SetVelocity(0.0f, 0.0f, 0.0f); // 0m/s
        LFParticle.SetAcceleration(0.0f, 0.0f, 0.0f);
        LFParticle.SetDamping(0.99f);
        LFParticle.SetPosition(leftFoot.position);

        RFParticle = new cyclone.Particle();
        RFParticle.SetMass(2.0f); // 2.0kg
        RFParticle.SetVelocity(0.0f, 0.0f, 0.0f); // 0m/s
        RFParticle.SetAcceleration(0.0f, 0.0f, 0.0f);
        RFParticle.SetDamping(0.99f);
        RFParticle.SetPosition(rightFoot.position);
    }

    /// <summary>
    /// update method for particles
    /// </summary>
    void ParticleUpdate()
    {
        // Find the duration of the last frame in seconds
        float duration = Time.fixedDeltaTime;//(float)TimingData::get().lastFrameDuration * 0.001f;
        if (duration <= 0.0f) return;//yield return null;

        // Update the physics of each particle in turn
        cyclone.Particle particle=new cyclone.Particle();
        for (int i = 0; i < 4; i++)//hands+feet=4
        {
            if (i == 0) particle = LHParticle;
            else if (i == 1) particle = RHParticle;
            else if (i == 2) particle = LFParticle;
            else if (i == 3) particle = RFParticle;

            // Run the physics
            particle.Integrate(duration);
        }

        //update particle GFX
        leftHand.position = new Vector3((float)LHParticle.position.x, (float)LHParticle.position.y, (float)LHParticle.position.z);
        rightHand.position = new Vector3((float)RHParticle.position.x, (float)RHParticle.position.y, (float)RHParticle.position.z);
        leftFoot.position = new Vector3((float)LFParticle.position.x, (float)LFParticle.position.y, (float)LFParticle.position.z);
        rightFoot.position = new Vector3((float)RFParticle.position.x, (float)RFParticle.position.y, (float)RFParticle.position.z);
    }


    /// <summary>
    /// changes status of hand movement variables
    /// </summary>
    /// <param name="changeRightHand">true if we should change right hand</param>
    /// <param name="newStatus"></param>
    /// <param name="time"></param>
    /// <returns></returns>
    IEnumerator ChangeHandMovementStatus(bool changeRightHand, bool newStatus, float time)
    {
        yield return new WaitForSecondsRealtime(time);

        if(changeRightHand)
        {
            isRightHandMoving = newStatus;
        }
        else
        {
            isLeftHandMoving = newStatus;
        }
    }

    /// <summary>
    /// activates handle holding mechanism of a hand after time
    /// </summary>
    /// <param name="activateRightHand">true if we want to activate right hand's handle holding mechanism</param>
    /// <param name="time"></param>
    /// <returns></returns>
    IEnumerator ActivateHandleHolding(bool activateRightHand,float time)
    {
        yield return new WaitForSecondsRealtime(time);
        
        if(activateRightHand)
        {
            //TODO DECISION increase hold distance
            //find if there is any handles we can hold nearby, hold the nearest
            //TODO update to cyclone overlap sphere & change handle hold positions colliders
            Collider[] handleHoldingPositions = Physics.OverlapSphere(rightHand.position, 0.2f, handleAttachPositions);
            if(handleHoldingPositions.Length==0)
            {
                //TODO drop the hand, return
                yield break;
            }
            Debug.Log("held handle with right hand, right hand speed.y="+RHParticle.GetVelocity().y);
            float nearestDistSqred = (handleHoldingPositions[0].transform.position - rightHand.position).sqrMagnitude;
            float nextDistSqred;
            int closestObjectIndex = 0;
            for (int i = 1; i < handleHoldingPositions.Length; i++)//don't need to re-calculate first distance sqred
            {
                nextDistSqred= (handleHoldingPositions[i].transform.position - rightHand.position).sqrMagnitude;
                if(nextDistSqred<nearestDistSqred)
                {
                    nearestDistSqred = nextDistSqred;
                    closestObjectIndex = i;
                }
            }
            //hold the holding position
            isRightHandAttached = true;
            RHParticle.SetPosition(handleHoldingPositions[closestObjectIndex].transform.position);
        }
        else//activate left hand
        {
            //TODO DECISION increase hold distance
            //find if there is any handles we can hold nearby, hold the nearest
            //TODO update to cyclone overlap sphere & change handle hold positions colliders
            Collider[] handleHoldingPositions = Physics.OverlapSphere(leftHand.position, 0.4f,handleAttachPositions);
            if (handleHoldingPositions.Length == 0)
            {
                //TODO drop the hand, return
                yield break;
            }
            Debug.Log("held handle with left hand, left hand particle speed.y=" + LHParticle.GetVelocity().y + "lefthand object& particle position.y="+
                leftHand.position.y+"   "+LHParticle.position.y);
            float nearestDistSqred = (handleHoldingPositions[0].transform.position - leftHand.position).sqrMagnitude;
            float nextDistSqred;
            int closestObjectIndex = 0;
            for (int i = 1; i < handleHoldingPositions.Length; i++)//don't need to re-calculate first distance sqred
            {
                nextDistSqred = (handleHoldingPositions[i].transform.position - leftHand.position).sqrMagnitude;
                if (nextDistSqred < nearestDistSqred)
                {
                    nearestDistSqred = nextDistSqred;
                    closestObjectIndex = i;
                }
            }
            //hold the holding position
            isLeftHandAttached = true;
            LHParticle.SetPosition(handleHoldingPositions[closestObjectIndex].transform.position);
        }
    }

    private void FallDown()
    {
        isLeftHandAttached = false;
        isRightHandAttached = false;
        isRightHandMoving = true;
        isLeftHandMoving = true;

        player.transform.position += Vector3.down * 10 * Time.deltaTime;
        RFParticle.SetAcceleration(new cyclone.MyVector3(0, -10, 0));
        RHParticle.SetAcceleration(new cyclone.MyVector3(0, -10, 0));
        LFParticle.SetAcceleration(new cyclone.MyVector3(0, -10, 0));
        LHParticle.SetAcceleration(new cyclone.MyVector3(0, -10, 0));
        //playerCamSetup.position += new Vector3(0, -10*Time.deltaTime, 0);

    }


}
