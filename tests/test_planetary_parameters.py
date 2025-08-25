import pytest

from exo_finder.data_pipeline.generation.planetary_parameters import PlanetaryParameters


class TestPlanetaryParams:
    @pytest.fixture()
    def sample_parameters(self) -> PlanetaryParameters:
        return PlanetaryParameters(
            period_d=10.0,
            transit_midpoint_d=5.0,
            planet_radius_solrad=0.01,
            star_radius_solrad=1.0,
            semi_major_axis_solrad=20.0,
            planet_mass_solmass=0.001,
            inclination_deg=89.0,
            eccentricity=0.1,
            argument_of_periastron_deg=90.0,
            limb_darkening_c1=0.4,
            limb_darkening_c2=0.2,
        )

    def test_parameter_count(self, sample_parameters: PlanetaryParameters):
        numpy_parameters = sample_parameters.to_numpy()
        assert len(numpy_parameters) == sample_parameters.parameter_count()
