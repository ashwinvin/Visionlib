.PHONY: test package clean docs

all: package

test:
	@python3 -m unittest discover -s test_face -t test

package:
	@echo "Making package"
	@python3 setup.py sdist bdist_wheel
	@cd dist 
	@echo "Installing package" 
	@pip3 install --force  dist/visionlib-0.6-py3-none-any.whl

docs:
	@sphinx-apidoc -f -o docs/ visionlib
	$(MAKE) -C docs html

clean:
	@find . -name '*.pyc' -exec rm --force {} +
	@find . -name '__pycache__' -exec rm -rf --force {} +
	@find . -name 'build' -exec rm -rf --force {} +
	@find . -name 'dist' -exec rm -rf --force {} +
	@find . -name 'visionlib.egg-info' -exec rm -rf --force {} +
