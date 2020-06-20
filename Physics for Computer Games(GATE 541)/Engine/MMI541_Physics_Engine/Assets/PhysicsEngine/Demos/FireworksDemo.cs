using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace cyclone
{

    public class FireworksDemo : MonoBehaviour
    {
        public bool isFireworksDemoActive = true;
        [SerializeField] GameObject fireworkPrefab;//used for GFX of fireworks
        GameObject[] fireworkGFX = new GameObject[10*maxFireworks];//used for updating gfx of fireworks
        /**
        * Holds the maximum number of fireworks that can be in use.
        */
        static uint maxFireworks = 1024;

        /** Holds the firework data. */
        Firework[] fireworks =new Firework[maxFireworks];

        /** Holds the index of the next firework slot to use. */
        uint nextFirework = 0;

        /** And the number of rules. */
        static uint ruleCount = 9;

        /** Holds the set of rules. */
        FireworkRule[] rules=new FireworkRule[ruleCount];

        /** Creates a new demo object. */
        public FireworksDemo()
        {
        }

        public FireworksDemo(uint nextFirework)
        {
            if (nextFirework==0)
            {
                // Make all shots unused
                foreach (Firework f in fireworks)
                {
                    f.type = 0;
                }

                // Create the firework types
                initFireworkRules();
            }
        }

        void initFireworkRules()
        {
            // Go through the firework types and create their rules.
            rules[0].Init(2);
            rules[0].SetParameters(
                1, // type
                0.5f, 1.4f, // age range
                new MyVector3(-5, 25, -5), // min velocity
                new MyVector3(5, 28, 5), // max velocity
                0.1 // damping
                );
            rules[0].payloads[0].Set(3, 5);
            rules[0].payloads[1].Set(5, 5);

            rules[1].Init(1);
            rules[1].SetParameters(
                2, // type
                0.5f, 1.0f, // age range
                new MyVector3(-5, 10, -5), // min velocity
                new MyVector3(5, 20, 5), // max velocity
                0.8 // damping
                );
            rules[1].payloads[0].Set(4, 2);

            rules[2].Init(0);
            rules[2].SetParameters(
                3, // type
                0.5f, 1.5f, // age range
                new MyVector3(-5, -5, -5), // min velocity
                new MyVector3(5, 5, 5), // max velocity
                0.1 // damping
                );

            rules[3].Init(0);
            rules[3].SetParameters(
                4, // type
                0.25f, 0.5f, // age range
                new MyVector3(-20, 5, -5), // min velocity
                new MyVector3(20, 5, 5), // max velocity
                0.2 // damping
                );

            rules[4].Init(1);
            rules[4].SetParameters(
                5, // type
                0.5f, 1.0f, // age range
                new MyVector3(-20, 2, -5), // min velocity
                new MyVector3(20, 18, 5), // max velocity
                0.01 // damping
                );
            rules[4].payloads[0].Set(3, 5);

            rules[5].Init(0);
            rules[5].SetParameters(
                6, // type
                3, 5, // age range
                new MyVector3(-5, 5, -5), // min velocity
                new MyVector3(5, 10, 5), // max velocity
                0.95 // damping
                );

            rules[6].Init(1);
            rules[6].SetParameters(
                7, // type
                4, 5, // age range
                new MyVector3(-5, 50, -5), // min velocity
                new MyVector3(5, 60, 5), // max velocity
                0.01 // damping
                );
            rules[6].payloads[0].Set(8, 10);

            rules[7].Init(0);
            rules[7].SetParameters(
                8, // type
                0.25f, 0.5f, // age range
                new MyVector3(-1, -1, -1), // min velocity
                new MyVector3(1, 1, 1), // max velocity
                0.01 // damping
                );

            rules[8].Init(0);
            rules[8].SetParameters(
                9, // type
                3, 5, // age range
                new MyVector3(-15, 10, -5), // min velocity
                new MyVector3(15, 15, 5), // max velocity
                0.95 // damping
                );
            // ... and so on for other firework types ...
        }


        void Create(uint type, Firework parent)
        {
            // Get the rule needed to create this firework
            FireworkRule rule = rules[type-1];

            // Create the firework
            rule.Create(fireworks[nextFirework], parent);
            fireworkGFX[nextFirework] = Instantiate(fireworkPrefab);

            // Increment the index for the next firework
            nextFirework = (nextFirework + 1) % maxFireworks;
        }

        void Create(uint type, uint number, Firework parent)
        {
            for (uint i = 0; i < number; i++)
            {
                Create(type, parent);
            }
        }


        void FireWorksUpdate()
        {
            // Find the duration of the last frame in seconds
            float duration = Time.fixedDeltaTime;//(float)TimingData::get().lastFrameDuration * 0.001f;
            if (duration <= 0.0f) return;//yield return null;

            //update each firework in turn
            Firework firework;
            for(int fireW=0;fireW<maxFireworks;fireW++)
            {
                firework = fireworks[fireW];
                // Check if we need to process this firework.
                if (firework.type > 0)
                {
                    // Does it need removing?
                    if (firework.UpdateFirework(duration))
                    {
                        //Debug.Log("firework" + fireW + " velocity.x=" + firework.GetVelocity().x);
                        // Find the appropriate rule
                        FireworkRule rule = rules[ (firework.type - 1)];

                        // Delete the current firework (this doesn't affect its
                        // position and velocity for passing to the create function,
                        // just whether or not it is processed for rendering or
                        // physics.
                        firework.type = 0;

                        //erase fireworks gfx
                        Destroy(fireworkGFX[fireW]);

                        // Add the payload
                        for (uint i = 0; i < rule.payloadCount; i++)
                        {
                            cyclone.FireworkRule.Payload payload = rule.payloads[i];
                            Create(payload.type, payload.count, firework);

                            Debug.Log("creating payload of type" + payload.type);
                        }
                    }
                    else
                    {
                        //update fireworks GFX
                        fireworkGFX[fireW].transform.position = new Vector3((float)firework.GetPosition().x, (float)firework.GetPosition().y, (float)firework.GetPosition().z);
                        if (fireW==1 ||fireW==2) Debug.Log(fireW+"nd firework's position.x=" + (float)firework.GetPosition().x +
                            "velocity.x="+(float)firework.GetVelocity().x);
                    }
                }
            }

            //yield return new WaitForEndOfFrame();//Application::update();
            //StartCoroutine(FireWorksUpdate());//restart fireworks physics
        }



        private void Start()
        {

            for (int i = 0; i < maxFireworks; i++)
            {
                Firework firework = new Firework();
                fireworks[i] = firework;
            }
            for (int i = 0; i < ruleCount; i++)
            {
                FireworkRule rule = new FireworkRule();
                //for (int j = 0; j < ; j++)
                //{

                //}
                rules[i] = rule;
            }
            initFireworkRules();
            

            //StartCoroutine(FireWorksUpdate());//start fireworks physics
        }

        private void FixedUpdate()
        {
            FireWorksUpdate();
        }

        private void Update()
        {
            //get inputs if fireworks demo active
            if (isFireworksDemoActive == false) return;

            if(Input.GetKeyDown("1"))
            {
                Create(1, 1, null);
            }
            else if(Input.GetKeyDown("2"))
            {
                Create(2, 1, null);
            }
            else if (Input.GetKeyDown("3"))
            {
                Create(3, 1, null);
            }
            else if (Input.GetKeyDown("4"))
            {
                Create(4, 1, null);
            }
            else if (Input.GetKeyDown("5"))
            {
                Create(5, 1, null);
            }
            else if (Input.GetKeyDown("6"))
            {
                Create(6, 1, null);
            }
            else if (Input.GetKeyDown("7"))
            {
                Create(7, 1, null);
            }
            else if (Input.GetKeyDown("8"))
            {
                Create(8, 1, null);
            }
            else if (Input.GetKeyDown("9"))
            {
                Create(9, 1, null);
            }
        }
    }


    
    /**
        * Fireworks are particles, with additional data for rendering and
        * evolution.
        */
    public class Firework : Particle
    {
        /** Fireworks have an integer type, used for firework rules. */
        public uint type;

        /**
         * The age of a firework determines when it detonates. Age gradually
         * decreases, when it passes zero the firework delivers its payload.
         * Think of age as fuse-left.
         */
        public double age;

        public Firework()
        {
        }

        /**
         * Updates the firework by the given duration of time. Returns true
         * if the firework has reached the end of its life and needs to be
         * removed.
         */
        public bool UpdateFirework(double duration)
        {
            // Update our physical state
            Integrate(duration);

            // We work backwards from our age to zero.
            age -= duration;
            return (age < 0) || (position.y < 0);
        }
    }

    /**
    * Firework rules control the length of a firework's fuse and the
    * particles it should evolve into.
    */
    public class FireworkRule
    {
        /** The type of firework that is managed by this rule. */
        uint type;

        /** The minimum length of the fuse. */
        double minAge;

        /** The maximum legnth of the fuse. */
        double maxAge;

        /** The minimum relative velocity of this firework. */
        MyVector3 minVelocity;

        /** The maximum relative velocity of this firework. */
        MyVector3 maxVelocity;

        /** The damping of this firework type. */
        double damping;

        /**
         * The payload is the new firework type to create when this
         * firework's fuse is over.
         */
        public class Payload
        {
            /** The type of the new particle to create. */
            public uint type;

            /** The number of particles in this payload. */
            public uint count;

            public Payload()
            {
            }
            
            /** Sets the payload properties in one go. */
            public void Set(uint type, uint count)
            {
                this.type = type;
                this.count = count;
            }

            
        }

        /** The number of payloads for this firework type. */
        public int payloadCount;

        /** The set of payloads. */
        public Payload[] payloads;

        //default constructor
        public FireworkRule()//:payloadCount(0),payloads(NULL)
        {
            payloadCount = 0;
            payloads = null;
        }

        public void Init(int payloadCount)
        {
            this.payloadCount = payloadCount;
            payloads = new Payload[payloadCount];
            for (int i = 0; i < payloadCount; i++)
            {
                Payload payload = new Payload();
                payloads[i] = payload;
            }
        }

        //c# has garbage collection
        //~FireworkRule()
        //{
        //    if (payloads != NULL) delete[] payloads;
        //}

        /**
         * Set all the rule parameters in one go.
         */
        public void SetParameters(uint type, double minAge, double maxAge,
            MyVector3 minVelocity, MyVector3 maxVelocity,
        double damping)
        {
            this.type = type;
            this.minAge = minAge;
            this.maxAge = maxAge;
            this.minVelocity = minVelocity;
            this.maxVelocity = maxVelocity;
            this.damping = damping;
        }

        /**
         * Creates a new firework of this type and writes it into the given
         * instance. The optional parent firework is used to base position
         * and velocity on.
         */
        public void Create(Firework firework, Firework parent = null)
        {
            firework.type = type;
            firework.age = UnityEngine.Random.Range((float)minAge, (float)maxAge);

            MyVector3 vel = new MyVector3(0, 0, 0);
            if (parent != null)
            {
                // The position and velocity are based on the parent.
                firework.SetPosition(parent.GetPosition());
                vel += parent.GetVelocity();
            }
            else
            {
                MyVector3 start = new MyVector3(0, 0, 0);
                int x = UnityEngine.Random.Range(0, 3);// - 1;//(int)crandom.randomInt(3) - 1;
                start.x = 5.0f * (double)x;
                firework.SetPosition(start);
            }

            vel += new MyVector3(UnityEngine.Random.Range((float)minVelocity.x, (float)maxVelocity.x),
                            UnityEngine.Random.Range((float)minVelocity.y, (float)maxVelocity.y),
                            UnityEngine.Random.Range((float)minVelocity.z, (float)maxVelocity.z));// crandom.randomVector(minVelocity, maxVelocity);

            firework.SetVelocity(vel);

            //if (parent != null) Debug.Log("child velocity.x=" + firework.GetVelocity().x);
            // We use a mass of one in all cases (no point having fireworks
            // with different masses, since they are only under the influence
            // of gravity).
            firework.SetMass(1);

            firework.SetDamping(damping);

            firework.SetAcceleration(new MyVector3(0, -9.81f, 0));//MyVector3::GRAVITY);

            //Debug.Log("firework || damping=" + firework.GetDamping() + " // velocity=" + firework.GetVelocity() + " // acceleration=" + firework.GetAcceleration() +
               // " // position=" + firework.GetPosition());

            firework.ClearAccumulator();
        }
    }


}
