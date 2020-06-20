public class SoftTire extends Tire
{
	public SoftTire()
	{
		this.speed = 350;
		this.degradation = 0;
	}
	
	@Override
	public void tick(TrackFeature feature)
	{
		this.degradation += feature.getFeatureTypeMultiplier() * feature.getRoughness() * 1.2;
		if (this.speed >= 100)
		{
			this.speed -= Math.min(75.0, degradation) * 0.25;
		}
	}
	
	@Override
	public Tire getPitStopTire()
	{
		return new MediumTire();
	}
}
