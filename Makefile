.PHONY: package clean

all: package

package:
	@echo "Making package"
	@python3 setup.py sdist bdist_wheel
	@cd dist 
	@echo "Installing package" 
	@pip3 install .

clean:
	@find . -name '*.pyc' -exec rm --force {} +
	@find . -name '__pycache__' -exec rm -rf --force {} +
	@find . -name 'build' -exec rm -rf --force {} +
	@find . -name 'dist' -exec rm -rf --force {} +
	@find . -name 'visionlib.egg-info' -exec rm -rf --force {} +
