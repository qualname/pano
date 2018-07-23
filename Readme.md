# Panorama keszito program

A program rendezetlen kepek sorozatabol keszit panorama kep(ek)et.


### Telepites

A telepites megkezdese elott rendelkezned kell `cmake`, `OpenCV` 3.4 (vagy ujabb), illetve a megfelelo OpenCV-contrib programokkal.  
Szukseg lesz tovabba egy C++11-et tamogato forditohoz is.  
A kod Linux alatt c++ es clang forditokkal lett tesztelve. Elmeletileg windows alatt is kellene futnia, ha megsem nyiss egy issuet!  

A telepites:

```bash
mkdir build
cd build
cmake ..
```

### Hasznalat

A forditott binaris file (`a.out` modositatlan `CMakeLists.txt` file-lal) parameterul kepek neveit varja.  
Kimenete(i) az osszefuzott panoramakep(ek) `.png` kiterjesztessel.
