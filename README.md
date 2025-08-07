# ExoFinder
Deep Learning models for finding exoplanets with the transit method in TESS light curves

## WIP
This project is going through a major refactor to make it more maintainable, understandable, and easier to use, 
you can find the full code at [https://version.helsinki.fi/ccardin/exoplanet-finder](https://version.helsinki.fi/ccardin/exoplanet-finder)


## Secret Management

This project uses environment variables for managing sensitive information like API keys and credentials. To set up your secrets:

1. Create a `.env` in the project root
2. Edit the `.env` file and add your actual credentials:
   ```
   MAST_TOKEN=your_actual_token
   GAIA_USER=your_actual_username
   GAIA_PASSWORD=your_actual_password
   ```

3. The `.env` file is automatically ignored by git, so your secrets won't be committed to version control.

4. In your code, access secrets using the utility functions:
   ```python
   from exo_finder.utils.secrets import get_secret
   
   api_key = get_secret("API_KEY")
   ```
