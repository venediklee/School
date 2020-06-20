using System.Collections;
using System.Collections.Generic;
using UnityEngine;

//playtesting particle class
namespace cyclone
{
    public class ParticleDemo : MonoBehaviour
    {
        public bool isParticleDemoActive = false;

        [SerializeField] GameObject pistolParticlePrefab;//used for creating gfx of particles
        GameObject []particleGFX=new GameObject[ammoRounds];//used for updating gfx of particles

        float h, v;//horizontal, vertical forces to apply

        public enum ShotType
        {
            UNUSED = 0,
            PISTOL,
            ARTILLERY,
            FIREBALL,
            LASER,
            SPECIAL
        };

        /**
        * Holds a single ammunition round record.
        */
        [System.Serializable]
        public class AmmoRound
        {
            public cyclone.Particle particle;
            public cyclone.ParticleDemo.ShotType type = ShotType.UNUSED;
            //DEV_NOTE :: times are represented as floats not uints in my implementation
            public float startTime;

            public AmmoRound(Particle particle, ShotType type, float startTime)
            {
                this.particle = particle;
                this.type = type;
                this.startTime = startTime;
            }
        }

        /**
         * Holds the maximum number of  rounds that can be
         * fired.
         */
        const int ammoRounds = 16;

        /** Holds the particle data. */
        [SerializeField] AmmoRound[] ammo = new AmmoRound[ammoRounds];

        /** Holds the current shot type. */
        ShotType currentShotType=ShotType.PISTOL;

        private void Start()
        {
            for (int i = 0; i < ammoRounds; i++)
            {
                AmmoRound round = new AmmoRound(new Particle(), ShotType.UNUSED, 0f);
                ammo[i] = round;
            }
            //StartCoroutine(ParticleUpdate());
        }

        void Fire()
        {
            AmmoRound shot;
            //Find the first available round.
            for (int i = 0; ; i++)
            {
                shot = ammo[i];
                Debug.Log("i is " + i + "ammo[i] is null->" + (ammo[i] == null));
                if (ammo[i].type == ShotType.UNUSED)
                {
                    particleGFX[i] = Instantiate(pistolParticlePrefab);
                    break;
                }
                // If we didn't find a round, then exit - we can't fire.
                if (i == ammoRounds-1) return;
            }

            
            

            // Set the properties of the particle
            switch (currentShotType)
            {
                case ShotType.PISTOL:
                    shot.particle.SetMass(2.0f); // 2.0kg
                    shot.particle.SetVelocity(0.0f, 0.0f, 35.0f); // 35m/s
                    shot.particle.SetAcceleration(0.0f, -1.0f, 0.0f);
                    shot.particle.SetDamping(0.99f);
                    break;

                case ShotType.ARTILLERY:
                    shot.particle.SetMass(200.0f); // 200.0kg
                    shot.particle.SetVelocity(0.0f, 30.0f, 40.0f); // 50m/s
                    shot.particle.SetAcceleration(0.0f, -20.0f, 0.0f);
                    shot.particle.SetDamping(0.99f);
                    break;

                case ShotType.FIREBALL:
                    shot.particle.SetMass(1.0f); // 1.0kg - mostly blast damage
                    shot.particle.SetVelocity(0.0f, 0.0f, 10.0f); // 5m/s
                    shot.particle.SetAcceleration(0.0f, 0.6f, 0.0f); // Floats up
                    shot.particle.SetDamping(0.9f);
                    break;

                case ShotType.LASER:
                    // Note that this is the kind of laser bolt seen in films,
                    // not a realistic laser beam!
                    shot.particle.SetMass(0.1f); // 0.1kg - almost no weight
                    shot.particle.SetVelocity(0.0f, 0.0f, 100.0f); // 100m/s
                    shot.particle.SetAcceleration(0.0f, 0.0f, 0.0f); // No gravity
                    shot.particle.SetDamping(0.99f);
                    break;

                case ShotType.SPECIAL://used for testing fields of particle
                    shot.particle.SetMass(1.0f);
                    shot.particle.SetVelocity(30.0f, 0.0f, 3.0f);
                    shot.particle.SetAcceleration(0.0f, 0.0f, 0.0f);
                    shot.particle.AddForce(new MyVector3(-300f, 0.0f, 0.0f));
                    shot.particle.SetDamping(0.99f);
                    break;

            }

            // Set the data common to all particle types
            shot.particle.SetPosition(0.0f, 1.5f, 0.0f);
            shot.startTime = Time.time;//TimingData::get().lastFrameTimestamp;
            shot.type = currentShotType;

            // Clear the force accumulators
            shot.particle.ClearAccumulator();

        }

        //update method for particles
        void ParticleUpdate()
        {
            // Find the duration of the last frame in seconds
            float duration = Time.fixedDeltaTime;//(float)TimingData::get().lastFrameDuration * 0.001f;
            if (duration <= 0.0f) return;//yield return null;

            // Update the physics of each particle in turn
            AmmoRound shot;
            for (int i = 0; i < ammoRounds; i++)
            {
                shot = ammo[i];
                if (shot.type != ShotType.UNUSED)
                {

                    shot.particle.AddForce(new MyVector3(h,  v, 0));
                    // Run the physics
                    shot.particle.Integrate(duration);
                    // Check if the particle is now invalid
                    if (shot.particle.GetPosition().y < 0.0f ||
                        shot.startTime + 5000 < Time.time || //TimingData::get().lastFrameTimestamp ||
                        shot.particle.GetPosition().z > 200.0f)
                    {
                        // We simply set the shot type to be unused, so the
                        // memory it occupies can be reused by another shot.
                        //Debug.Log("particle invalidated");
                        shot.type = ShotType.UNUSED;

                        //erase particle gfx
                        Destroy(particleGFX[i]);
                    }
                    else
                    {
                        //update particle GFX
                        particleGFX[i].transform.position = new Vector3((float)shot.particle.position.x, (float)shot.particle.position.y, (float)shot.particle.position.z);
                    }
                }
            }

            //yield return new WaitForEndOfFrame();//Application::update();
            //StartCoroutine(ParticleUpdate());//restart physics
        }

        private void Update()
        {
            //get inputs if particle demo is active
            if (isParticleDemoActive == false) return;
            

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
            if (Input.GetKeyDown("5"))
            {
                currentShotType = ShotType.SPECIAL;
                Debug.Log("currentShotType->" + currentShotType);
            }
            if (Input.GetButtonDown("Fire1"))
            {
                Fire();
            }

            if (Input.GetKeyDown("d")) h = 1f;
            else if (Input.GetKeyDown("a")) h = -1f;
            else h = 0;
            if (Input.GetKeyDown("w")) v = 1f;
            else if (Input.GetKeyDown("s")) v = -1f;
            else v = 0;

        }
        private void FixedUpdate()
        {
            ParticleUpdate();
        }
    }
}
