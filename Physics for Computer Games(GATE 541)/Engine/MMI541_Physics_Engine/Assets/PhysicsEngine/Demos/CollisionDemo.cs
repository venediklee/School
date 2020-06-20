using System.Collections;
using System.Collections.Generic;
using UnityEngine;


namespace cyclone
{
    using real = System.Double;

    enum ShotType
    {
        UNUSED = 0,
        PISTOL,
        ARTILLERY,
        FIREBALL,
        LASER
    };


    class AmmoRound : CollisionSphere
    {

        public ShotType type;
        public float startTime;

        public AmmoRound()
        {
            body = new cyclone.RigidBody();
        }

        ~AmmoRound()
        {
            body = null;
        }

        /** Sets the box to a specific location. */
        public void setState(ShotType shotType)
        {
            type = shotType;

            // Set the properties of the particle
            switch (type)
            {
                case ShotType.PISTOL:
                    body.setMass(1.5f);
                    body.setVelocity(0.0f, 0.0f, 20.0f);
                    body.setAcceleration(0.0f, -0.5f, 0.0f);
                    body.setDamping(0.99f, 0.8f);
                    radius = 0.2f;
                    break;

                case ShotType.ARTILLERY:
                    body.setMass(200.0f); // 200.0kg
                    body.setVelocity(0.0f, 30.0f, 40.0f); // 50m/s
                    body.setAcceleration(0.0f, -21.0f, 0.0f);
                    body.setDamping(0.99f, 0.8f);
                    radius = 0.4f;
                    break;

                case ShotType.FIREBALL:
                    body.setMass(4.0f); // 4.0kg - mostly blast damage
                    body.setVelocity(0.0f, -0.5f, 10.0); // 10m/s
                    body.setAcceleration(0.0f, 0.3f, 0.0f); // Floats up
                    body.setDamping(0.9f, 0.8f);
                    radius = 0.6f;
                    break;

                case ShotType.LASER:
                    // Note that this is the kind of laser bolt seen in films,
                    // not a realistic laser beam!
                    body.setMass(0.1f); // 0.1kg - almost no weight
                    body.setVelocity(0.0f, 0.0f, 100.0f); // 100m/s
                    body.setAcceleration(0.0f, 0.0f, 0.0f); // No gravity
                    body.setDamping(0.99f, 0.8f);
                    radius = 0.2f;
                    break;
            }

            body.setCanSleep(false);
            body.setAwake();

            Matrix3 tensor = new Matrix3();
            real coeff = 0.4f * body.getMass() * radius * radius;
            tensor.setInertiaTensorCoeffs(coeff, coeff, coeff);
            body.setInertiaTensor(tensor);

            // Set the data common to all particle types
            body.setPosition(0.0f, 1.5f, 0.0f);
            startTime = Time.time;//TimingData::get().lastFrameTimestamp;

            // Clear the force accumulators
            body.calculateDerivedData();
            calculateInternals();
        }
    };



    class Box : CollisionBox
    {

        public Box()
        {
            body = new cyclone.RigidBody();
        }

        ~Box()
        {
            body = null;
        }


        /** Sets the box to a specific location. */
        public void setState(real z)
        {
            body.setPosition(0, 3, z);
            body.setOrientation(1, 0, 0, 0);
            body.setVelocity(0, 0, 0);
            body.setRotation(new MyVector3(0, 0, 0));
            halfSize = new MyVector3(1, 1, 1);

            real mass = halfSize.x * halfSize.y * halfSize.z * 8.0f;
            body.setMass(mass);

            Matrix3 tensor = new Matrix3();
            tensor.setBlockInertiaTensor(halfSize, mass);
            body.setInertiaTensor(tensor);

            body.setLinearDamping(0.95f);
            body.setAngularDamping(0.8f);
            body.clearAccumulators();
            body.setAcceleration(0, -10.0f, 0);

            body.setCanSleep(false);
            body.setAwake();

            body.calculateDerivedData();
            calculateInternals();
        }
    };


    /**
 * The main demo class definition.
 */
    class CollisionDemo: MonoBehaviour //: RigidBodyApplication
    {
        [SerializeField] GameObject pistolParticlePrefab;//used for creating gfx of particles
        GameObject[] particleGFX = new GameObject[ammoRounds];//used for updating gfx of particles
        GameObject[] boxGFX = new GameObject[boxes];

        public bool isCollisionDemoActive;

        /** Holds the array of contacts. */
        CollisionData cData=new CollisionData();
        /** Holds the maximum number of contacts. */
        public readonly static uint maxContacts = 256;

        /** Holds the contact resolver. */
        ContactResolver resolver;

        /**
         * Holds the maximum number of  rounds that can be
         * fired.
         */
        readonly static uint ammoRounds = 256;

        /** Holds the particle data. */
        AmmoRound[] ammo = new AmmoRound[ammoRounds];

        /**
        * Holds the number of boxes in the simulation.
        */
        readonly static uint boxes = 2;

        /** Holds the box data. */
        Box[] boxData = new Box[boxes];

        /** Holds the current shot type. */
        ShotType currentShotType;

        /** Resets the position of all the boxes and primes the explosion. */
        public virtual void reset()
        {
            // Make all shots unused
            AmmoRound shot = new AmmoRound();
            for (int i = 0; i < ammoRounds; i++)
            {
                shot = ammo[i];
                if (shot == null) continue;
                shot.type = ShotType.UNUSED;
            }

            // Initialise the box
            real z = 20.0f;
            Box box = new Box();
            for (int i = 0; i < boxes; i++)
            {
                box = boxData[i];
                if (box == null) continue;
                box.setState(z);
                z += 90.0f;
                boxGFX[i] = Instantiate(pistolParticlePrefab);
                boxGFX[i].name = "boxGFX" + i.ToString();
                boxGFX[i].transform.position = MyVector3.ConvertToVector3(box.body.getPosition());
            }
        }

        /** Build the contacts for the current situation. */
        public virtual void generateContacts()
        {
            // Create the ground plane data
            cyclone.CollisionPlane plane = new CollisionPlane();
            plane.direction = new MyVector3(0, 1, 0);
            plane.offset = 0;

            // Set up the collision data structure
            cData = new CollisionData();
            cData.contacts = new Contact();
            cData.reset(maxContacts);
            cData.friction = (real)0.9;
            cData.restitution = (real)0.1;
            cData.tolerance = (real)0.1;

            // Check ground plane collisions
            Box box = new Box();
            for (int i = 0; i < boxes; i++)
            {
                box = boxData[i];
                if (!cData.hasMoreContacts()) return;
                CollisionDetector.boxAndHalfSpace(box, plane, cData);


                // Check for collisions with each shot
                AmmoRound shot = new AmmoRound();
                for (int j = 0; j < ammoRounds; j++)
                {
                    shot = ammo[j];
                    if (shot.type != ShotType.UNUSED)
                    {
                        if (!cData.hasMoreContacts()) return;

                        // When we get a collision, remove the shot
                        if (CollisionDetector.boxAndSphere(box, shot, cData) > 0)
                        {
                            shot.type = ShotType.UNUSED;
                        }
                    }
                }
            }

            // NB We aren't checking box-box collisions.
        }

        /** Processes the objects in the simulation forward in time. */
        public virtual void updateObjects(real duration)
        {
            // Update the physics of each particle in turn
            AmmoRound shot = new AmmoRound();
            for (int i = 0; i < ammoRounds; i++)
            {
                shot = ammo[i];
                if (shot.type != ShotType.UNUSED)
                {
                    // Run the physics
                    shot.body.integrate(duration);
                    shot.calculateInternals();

                    // Check if the particle is now invalid
                    if (shot.body.getPosition().y < 0.0f ||
                        shot.startTime + 5000 < Time.time || // TimingData::get().lastFrameTimestamp ||
                        shot.body.getPosition().z > 200.0f)
                    {
                        // We simply set the shot type to be unused, so the
                        // memory it occupies can be reused by another shot.
                        shot.type = ShotType.UNUSED;

                        Destroy(particleGFX[i]);
                    }
                    else
                    {
                        //update particle GFX
                        particleGFX[i].transform.position = new Vector3((float)shot.body.getPosition().x, (float)shot.body.getPosition().y, (float)shot.body.getPosition().z);
                    }
                }
            }

            // Update the boxes
            Box box = new Box();
            for (int i = 0; i < boxes; i++)
            {
                box = boxData[i];
                // Run the physics
                box.body.integrate(duration);
                box.calculateInternals();
                //Debug.Assert(boxGFX[i] != null, "box gfx is null");
                //Debug.Assert(box != null, "box is null");
                //Debug.Assert(box.body != null, "box rigidbody is null");
                //Debug.Assert(box.body.getPosition() != null, "box rigidbody position is null");
                boxGFX[i].transform.position = MyVector3.ConvertToVector3(box.body.getPosition());
            }
        }

        /** Dispatches a round. */
        void fire()
        {
            // Find the first available round.
            AmmoRound shot = new AmmoRound();
            for (int i = 0; i < ammoRounds; i++)
            {
                shot = ammo[i];
                if (ammo[i].type == ShotType.UNUSED)
                {
                    particleGFX[i] = Instantiate(pistolParticlePrefab);
                    particleGFX[i].name = "particleGFX" + i.ToString();
                    break;
                }

                // If we didn't find a round, then exit - we can't fire.
                if (i == ammoRounds - 1) return;
            }


            // Set the shot
            shot.setState(currentShotType);

        }


        /** Creates a new demo object. */
        public CollisionDemo()
        {
            currentShotType = ShotType.LASER;
            //pauseSimulation = false;
            reset();
        }

        private void Update()
        {
            //get inputs if collision demo is active
            if (isCollisionDemoActive == false) return;


            if (Input.GetKeyDown("1"))
            {
                currentShotType = ShotType.PISTOL;
                Debug.Log("currentShotType->" + currentShotType);
            }
            if (Input.GetKeyDown("2"))
            {
                currentShotType = ShotType.ARTILLERY;
                Debug.Log("currentShotType->" + currentShotType);
            }
            if (Input.GetKeyDown("3"))
            {
                currentShotType = ShotType.FIREBALL;
                Debug.Log("currentShotType->" + currentShotType);
            }
            if (Input.GetKeyDown("4"))
            {
                currentShotType = ShotType.LASER;
                Debug.Log("currentShotType->" + currentShotType);
            }
            if(Input.GetButtonDown("Fire1"))
            {
                fire();
            }
        }

        private void FixedUpdate()
        {
            // Update the objects
            updateObjects(Time.fixedDeltaTime);

            // Perform the contact generation
            generateContacts();

            // Resolve detected contacts
            resolver.resolveContacts(
                cData.contactArray,
                cData.contactCount,
                Time.fixedDeltaTime
                );
        }
        private void Start()
        {
            resolver= new ContactResolver(maxContacts * 8);

            for (int i = 0; i < ammoRounds; i++)
            {
                AmmoRound round = new AmmoRound();
                ammo[i] = round;
            }
            for(int i=0; i<boxes; i++)
            {
                Box box = new Box();
                boxData[i] = box;
                //GameObject gfx = Instantiate(pistolParticlePrefab);
                //gfx.name = "box gfx" + i.ToString();
                //boxGFX[i] = gfx;
            }
            reset();
        }

    };


}
