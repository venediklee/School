import java.lang.invoke.MethodHandleInfo;

public class HardTire extends Tire
{
	public HardTire()
	{
		this.speed = 275;
		this.degradation = 0;
	}
	
	@Override
	public void tick(TrackFeature feature)
	{
		this.degradation += feature.getFeatureTypeMultiplier() * feature.getRoughness();
		if (this.speed >= 100)
		{
			this.speed -= Math.min(75.0, degradation) * 0.25;
		}
	}
	
	@Override
	public Tire getPitStopTire()
	{
		return new SoftTire();
	}
}
