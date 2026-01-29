# Publishing to PyPI - Quick Guide

This guide will help you publish DimViz to PyPI.

## Prerequisites

1. **Create PyPI Account**
   - Sign up at https://pypi.org/account/register/
   - Verify your email

2. **Install Build Tools**
   ```bash
   pip install build twine
   ```

3. **Update Package Information**
   - Edit `setup.py` and `pyproject.toml`
   - Update author name and email
   - Update GitHub URLs
   - Ensure version number is correct

## Pre-Release Checklist

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Code is formatted: `black dimviz tests examples`
- [ ] Documentation is up to date
- [ ] README.md is complete and accurate
- [ ] CHANGELOG.md is updated
- [ ] Version number is incremented
- [ ] LICENSE file is present
- [ ] No sensitive information in code

## Building the Package

### 1. Clean Previous Builds
```bash
rm -rf build/ dist/ *.egg-info
```

### 2. Build Distribution Files
```bash
python -m build
```

This creates:
- `dist/dimviz-0.1.0-py3-none-any.whl` (wheel)
- `dist/dimviz-0.1.0.tar.gz` (source distribution)

### 3. Check the Build
```bash
twine check dist/*
```

## Testing the Package Locally

Install and test your package locally before publishing:

```bash
# Install from local build
pip install dist/dimviz-0.1.0-py3-none-any.whl

# Test it works
python -c "from dimviz import DimViz; print('Success!')"

# Run a quick test
python examples/basic_usage.py
```

## Publishing to Test PyPI (Recommended First Step)

Test PyPI is a separate instance for testing package uploads.

### 1. Create Test PyPI Account
Sign up at https://test.pypi.org/account/register/

### 2. Upload to Test PyPI
```bash
twine upload --repository testpypi dist/*
```

Enter your Test PyPI credentials when prompted.

### 3. Test Installation from Test PyPI
```bash
pip install --index-url https://test.pypi.org/simple/ dimviz
```

### 4. Verify Everything Works
```bash
python -c "from dimviz import DimViz; print('Test PyPI Success!')"
```

## Publishing to PyPI (Production)

Once you've verified everything works on Test PyPI:

### 1. Upload to PyPI
```bash
twine upload dist/*
```

Enter your PyPI credentials when prompted.

### 2. Verify on PyPI
- Visit https://pypi.org/project/dimviz/
- Check that the README renders correctly
- Verify all metadata is correct

### 3. Test Installation
```bash
pip install dimviz
python -c "from dimviz import DimViz; print('PyPI Success!')"
```

## Using API Tokens (Recommended)

For better security, use API tokens instead of passwords:

### 1. Generate API Token
- Go to https://pypi.org/manage/account/token/
- Create a new token
- Copy the token (you'll only see it once!)

### 2. Create `.pypirc` File
```bash
cat > ~/.pypirc << EOF
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TEST_TOKEN_HERE
EOF

chmod 600 ~/.pypirc
```

### 3. Upload with Token
```bash
twine upload dist/*
# Credentials will be read from .pypirc
```

## Post-Release Steps

1. **Create Git Tag**
   ```bash
   git tag -a v0.1.0 -m "Release version 0.1.0"
   git push origin v0.1.0
   ```

2. **Create GitHub Release**
   - Go to your GitHub repository
   - Click "Releases" → "Create a new release"
   - Select the tag you just created
   - Add release notes from CHANGELOG.md
   - Publish the release

3. **Announce the Release**
   - Share on social media
   - Post in relevant communities
   - Update documentation sites

## Versioning Guide

Follow semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

Examples:
- `0.1.0` → `0.1.1`: Bug fix
- `0.1.0` → `0.2.0`: New feature
- `0.1.0` → `1.0.0`: First stable release or breaking change

## Updating Your Package

To release a new version:

1. Make your changes
2. Update version in `setup.py` and `pyproject.toml`
3. Update `CHANGELOG.md`
4. Commit changes
5. Build: `python -m build`
6. Upload: `twine upload dist/*`
7. Tag: `git tag -a vX.Y.Z -m "Release X.Y.Z"`
8. Push: `git push origin vX.Y.Z`

## Common Issues

### Issue: "File already exists"
**Solution**: You can't re-upload the same version. Increment the version number.

### Issue: "Invalid distribution file"
**Solution**: Run `twine check dist/*` to identify issues.

### Issue: "Authentication failed"
**Solution**: Verify your credentials or API token.

### Issue: "README doesn't render"
**Solution**: Validate your Markdown and ensure long_description_content_type is set correctly.

## Resources

- **PyPI Guide**: https://packaging.python.org/tutorials/packaging-projects/
- **Twine Documentation**: https://twine.readthedocs.io/
- **Semantic Versioning**: https://semver.org/
- **Test PyPI**: https://test.pypi.org/

## Quick Reference

```bash
# Complete release workflow
rm -rf build/ dist/ *.egg-info
python -m build
twine check dist/*
twine upload --repository testpypi dist/*  # Test first
twine upload dist/*                         # Then production
git tag -a v0.1.0 -m "Release 0.1.0"
git push origin v0.1.0
```

Good luck with your release! 🚀
