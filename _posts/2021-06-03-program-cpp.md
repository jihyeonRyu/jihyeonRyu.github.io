---
layout: post  
title: (컴퓨터공학과 전공 기초 시리즈) C++ 프로그래밍         
subtitle: Programming     
tags: [c++, cpp, programing language, computer science, basic knowledge]    
comments: true  
---  
기본적으로 알고 있어야하는 __*Basic*__ 한 내용만을 다룹니다.  
윤성우 열혈 c++과 [씹어먹는 C++](https://modoocode.com/135) 을 참조하였습니다. 

# 1. C언어 기반의 C++
* C++ 의 지역변수 선언: 함수 내의 어디든 삽입이 가능 
* Function Overloading
    * 함수 호출 시 전달되는 인자를 통해 호출하고자 하는 함수의 구분이 가능하기 때문에 동일한 이름의 함수 정의를 사용할 수 있다.
    * 매개변수의 선언이 달라야 한다 (자료형, 개수)
    * 반환형은 구분 기준이 될 수 없다. 
* 매개변수의 디폴트값
    * 디폴트 값은 함수의 선언 부분에만 표현하면 됨
    * 오른쪽 매개변수의 디폴트 값부터 채우는 형태로 정의 
* inline function
    * 매크로 함수
        * 장점: 일반 함수에 비해 실행 속도의 이점이 있음
        * 단점: 복잡한 함수를 정의하는데 한계가 있음 
    * 인라인 함수
        * 매크로 함수의 장점을 유지하되 단점을 제거함.
        * 매크로를 이용한 함수의 인라인화는 전처리기에서 처리되지만, inline 키워드를 통한 인라인화는 컴파일러에 의해 처리가 된다. 
  

```cpp
#define SQUARE(x) ( (x)*(x) ) // 매크로 함수 

template <typename T>
inline  T SQUARE (T x) // 인라인 함수 
{
    return x*x;
}
```

* namespace 
    * 특정 영역에 이름을 붙여주기 위한 문법적 요소
    * :: 범위 지정 연산자
    * 동일한 이름 공간에 정의된 함수를 호출할 때에는 이름 공간을 명시할 필요 없음
    * using을 이용한 이름공간 명시
    * 지역변수와 전역 변수의 이름이 같은 경우, 전역 변수는 지역변수에 의해 가려짐
        * 전역변수를 지칭하기 위해 ::val 형식 사용 가능 
    
# 2. C언어 기반의 C++ 2
* const의 의미

```cpp
const int num=10; // num의 상수화
const int * ptr1 = &val // ptr1을 이용해 val의 값을 변경할 수 없음
int * const ptr2 = &val // ptr2가 상수화 되어 주소를 변경할 수 없음
const int * const ptr3 = &val // ptr3가 상수화 되고, 이를 통해 val의 값을 변경할 수 없음
```

* 실행중인 프로그램의 메모리 공간
    * 데이터: 전역 변수가 저장되는 영역
    * 스택: 지역변수 및 매개변수가 저장되는 영역
    * 힙: malloc 함수 호출에 의해 프로그램이 실행되는 과정에서 동적으로 할당이 이루어지는 영역
    
* Call-by-Value & Call-by-Reference
    * Call-by-Value: 값을 인자로 전달하는 함수 호출 방식
    * Call-by-Reference: 주소 값은 인자로 전달하는 방식
        * 참조자를 이용할 경우 함수 호출문 만으로 인자 값이 변경될지 여부를 알 수 없다.
        * 이를 해결하기 위한 방법으로 const 사용
    
```cpp
void swapByValue(int num1, int num2)
{
    int tmp = num1;
    num1 = num2;
    num2 = tmp;
}

void swapByReference(int * ptr1, int * ptr2)
{
    int tmp=*ptr1;
    *ptr1 = *ptr2;
    *ptr2 = tmp;
}

void swapByReference(int &ref1, int &ref2)
{
    int tmp = ref1;
    ref1 = ref2;
    ref2 = tmp;
}

void happyFunc(const int &ref)
{
    std::cout << ref << std::endl;
}
```
* 반환형이 Reference Type
    * 지역변수를 참조형으로 반환하면 안됨
    
```cpp
int& funcOne(int &ref)
{
    ref++;
    return ref;
}

int funcTwo(int &ref)
{
    ref++;
    return ref;
}

int num1=1;
int &num2 = funcOne(num1);
int num3 = funcOne(num1);

int num2 = funcTwo(num1);
```

* 새로운 자료형 bool: true, false 
* 참조자(Reference)의 이해 
    * 자신이 참조하는 변수를 대신할 수 있는 또 하나의 이름 (별칭)
    * 참조자의 수에 제한이 없으며, 참조자를 대상으로 참조자를 선언할 수 있음 
    * 변수에 대해서만 선언 가능
    * 선언과 동시에 변수를 참조해야 함
    * 참조의 대상을 바꾸는 것이 불가능 
    * NULL로 초기화 하는 것도 불가능 
    
```cpp
int num1 = 10;
int *ptr1 = &num1;
int **ptr2 = &ptr1;

int &ref = num1;
int *(&pref1) = ptr1;
int **(&pref2) = ptr2;
```

* const 참조자 
    * 상수도 참조하는 것이 가능: 임시변수를 만들어서 참조자가 이를 참조하게 함
    

```cpp
const int &ref = 30;
```

* Malloc & free 을 대신하는 new & delete
    * 참조자 선언을 통해 포인터 연산 없이 힙 영역에 접근 가능
    

```cpp
#include <iostream>
#include <stdlib.h>

char* MakeStrAdr(int len)
{
    // char * str = (char*) malloc(sizeof(char)*len);
    char * str = new char[len];
    return str;
}

char * str = MakeStrAdr(10);
char &ref = *str;
ref = "Hello";
// free(str);
delete []str;

```

# 3. 클래스의 기본
## c++에서의 구조체

```c
struct Car{
    int carSpeed;
    int fuel;
};

typedef struct _Car2 {
    int carSpeed;
    int fuel;
} Car2;

```
* C에서 선언
> struct Car basicCar;  
> Car2 basicCar2;

* C++에서의 선언 
> Car basicCar;

* 선언과 초기화
> Car basicCar = {50, 500};

* 구조체 안의 함수 선언
    * 호출 시 .(dot) 을 통해서 변수, 함수 접근 가능

```cpp
Struct Car{
    int carSpeed;
    int fuel;
    
    void showCarSpeed();
};

void Car::showCarSpeed(){
    std::cout << carSpeed << std::endl;
}

Car basicCar = {10, 500}; // 구조체 선언과 동시에 초기화 
basicCar.showCarSpeed();
```

## Class와 Object
* 접근제어 지시자
    * Public: 어디서든 접근 가능
    * Protected: 상속 관계에 놓였을 때, 유도 클래스에서 접근 허용
    * Private: 클래스 내에서만 접근 허용
  
```cpp
class Car 
{
private:
    char * carID[100];
    int speed;
public:
    void showSpeed();
} ;

inline void Car::showSpeed(){
    std::cout << speed << std::endl;
}
```

* 클래스 기반의 두 가지 객체 생성 방법 
  * 일반적인 변수 선언 방식
  > SimpleClass sc;   
  
  * 동적 할당 방식
  > SimpleClass* sc = new SimpleClass;   

* 멤버 함수에 대해서 inline 함수 사용 가능 inline 함수는 클래스가 선언된 헤더파일에 함께 넣어야함

# 4. 클래스의 완성
## 정보은닉 

```cpp
class Point{
public:
    int x;
    int y;
};

Point p = {10, 20}; // 멤버 변수가 public이므로 선언과 동시에 초기화 가능 .

```

* const 함수
    * 이 함수 내에서는 멤버변수에 저장된 값을 변경하지 않겠다.
    * const 함수 내에서는 const 가 아닌 함수의 호출이 제한된다. 
    * const 참조자를 이용해서는 const 함수만 호출 가능하다. 
    
> int func() const;  

## 생성자와 소멸자
### 생성자 
* 클래스의 이름과 함수의 이름이 동일하다
* 반환형이 선언되어 있지 않으며, 실제로 반환하지 않는다.
* 객체 생성 시 딱 한번 호출된다.
* 오버로딩이 가능하다.
* 매개변수에 디폴트 값을 설정할 수 있다.
* 생성자는 반드시 한번 호출 된다. 

```cpp
class SimpleClass {
private:
    int num;
public:
    SimpleClass (int n) {
        num = n;
    }
};

SimpleClass sc(10);
SimpleClass sc(); // (x)
SimpleClass sc;  // (o)

SimpleClass *scPtr = new SimpleClass(10); 
SimpleClass *scPtr = new SimpleClass();  // (o)
SimpleClass *scPtr = new SimpleClass;  // (o)
```

* 멤버 이니셜라이저를 이용한 초기화
    * 선언과 동시에 초기화가 이루어짐
    * const 변수도 초기화 가능 (const 변수는 선언과 동시에 초기화 되어야함)
    * 참조자도 초기화 가능 (참조자는 선언과 동시에 초기화 되어야함)
    

```cpp
class SimpleClass{
private:
    int num1;
    const int num2; // const 변수
    int &num3; // 참조자
    
public:
    SimpleClass(int n1, int n2, int n3): num1(n1), num2(n2), num3(n3) {}
};
```

* 디폴트 생성자
    * 생성자를 정의 하지 않았을때만 디폴트 생성자가 생성됨 

### 소멸자
* 객체 소멸시 반드시 호출
* 클래스의 이름앞에 ~가 붙은 형태의 이름을 갖음
* 반환형이 선언되지 않으므로, 반환하지 않음
* 매개변수는 void 형으로 선언되어야 하므로, 오버로딩이 불가능하다.

```cpp
class SimpleClass{
private:
    char* name;
public:

    SimpleClass() {} // 생성자 

    SimpleClass(int len){ // 생성자 
        name = new char[len+1];
    }
    ~SimpleClass(){ // 소멸자 
        delete []name;
    }
}
```

## 클래스 배열 
### 객체 배열
> SimpleClass arr[len];  
> SimpleClass * arrPtr = new SimpleClass[len];  

* Simplelass 클래스가 len개 생성되어 배열을 이룸
* 배열을 선언하는 경우에도 생성자가 호출이 됨
  * 단, 인자를 전달하지 못하므로 디폴트 생성자 형태의 생성자가 정의되어 있어야 한다.
* 배열이 소멸할 때, 객체 소멸자가 호출 됨

### 객체 포인터 배열
> SimpleClass * arrPtr[len]; 

* SimpleCLass 라는 클래스의 주소를 len개 담을 배열을 생성함 
* 별도의 동적 할당을 통해서 배열에 클래스의 주소를 초기화함
* 포인터로 객체 멤버에 접근은 -> 를 사용 

```cpp
int len = 10;
SimpleClass * arrPtr[len];

for (int i=0; i<len; i++){
    arrPtr[i] = new SimpleClass(); // 동적 할당 
    arrPtr[i]->printName();
}
```

## this 포인터
* 객체 자기 자신의 주소 값을 의미

## Self-Reference
* 객체 자기 자신을 참조 

```cpp
class SimpleClass{
private:
    int num;
public:
    SimpleClass(int num){
        this->num = num;
    }
    SimpleClass& returnSelf() {
        return *this;
    }
};

SimpleClass sc(10);
SimpleClass& ref = sc.returnSelf();

```

# 5. 복사 생성자 

```cpp
class SimpleClass {
private:
    int num1;
    int num2;
public:
    SimpleClass(int n1, int n2): num1(n1), num2(n2) {}; // 생성자 
    SimpleClass(SimpleClass& copy): num1(copy.num1), num2(copy.num2){}; // 복사 생성자 
}
SimpleClass sc1(10, 20); // 객체 생성 
SimpleClass sc2 = s1; // 1. 묵시적 변환이 일어남 
SimpleClass sc2(s1); // 2. 복사 생성자 호출 

```

* 복사 생성자를 정의하지 않으면, 맴버 대 멤버 복사를 진행하는 디폴트 복사 생성자가 자동으로 생성된다.
* 묵시적 호출을 허용하지 않으려면 explicit 키워드를 사용한다.
* 매개변수에 참조자 &를 사용하지 않으면 무한루프에 빠짐. (Call by value = 복사 생성자 호출)

```cpp
explicit SimpleClass(SimpleClass& copy): num1(copy.num1), num2(copy.num2) { };
```

## 깊은 복사와 얕은 복사
* 디폴트 복사 생성자는 멤버 대 멤버의 복사를 진행 (얕은 복사)
* 이런 경우 힙의 메모리 공간을 참조하는 경우 문제
    * 복사된 객체도 동시에 참조하는 문제 생김 (주소를 복사하기 때문에 같은 공간을 가리킴)
    * 객체 소멸과정에서 문제가 생김 (하나가 소멸하면 다른 하나는 쓰레기 값을 가리키게 됨)
    * 깊은 복사가 필요 
  
```cpp
class SimpleClass {
private:
    int num;
    char * name;
public:
    SimpleClass(int num, char * name){
        this->num = num;
        this->name = new char[strlen(name)+1];
        strcopy(this->name, name); 
    }
    
    SimpleClass(SimpleClass & copy){
        num = copy.num;
        name = new char[strlen(copy.name)+1]; // 깊은 복사 수행 
        strcopy(name, copy.name);
    }
    
    ~SimpleClass(){
        delete []name;
    }
}
```

* 복사 생성자의 호출 시점
    * 기존에 생성된 객체를 이용해 새로운 객체를 초기화 하는 경우
    * Call-by-value 방식의 함수 호출 과정에서 객체를 인자로 전달하는 경우
    * 객체를 반환하되, 참조형으로 반환하지 않는 경우  (임시객체 생성)
      * 임시객체는 다음 행으로 넘어가면 바로 소멸
      > get_temporary(300).ShowTemporary();  
      
      * 참조자에 의해 참조되는 임시 객체는 바로 소멸되지 않는다
      > Temporary &temp = get_temporary(300);  
      
# 6. Friend와 Static 그리고 Const
## const

* const 객체 
    * const 객체로 선언하면 const 멤버 함수만 호출 가능하다

```cpp
class SimpleClass{
private:
    int num;
public:
    SimpleClass(int n):num(n){}
    vois showNum() const { // const 멤버 함수 
        std::cout << num << std::endl;
    }
    void getNum() {
        return num;
    }
}

const SimpleClass sc(10); // const 객체로 생성  
sc.getNum(); // (x) 일반 멤버 함수 호출 불가 
sc.showNum(); // const 멤버 함수 호출 
```

* const 함수 오버로딩
> void SimpleFunc() { ... }   
> void SimpleFunc() const { ... }  

## friend
### 클래스의 friend 선언 
* A 클래스가 B 클래스에 대해서 friend 선언을 하면, B 클래스는 A 클래스의 private 멤버에 직접 접근이 가능하다. (반대는 성립 안됨) 
* friend 선언은 private, public 어디에 와도 상관 없다

```cpp
class B;

class A {
private:
    int a_num;
    friend class B;
    ...
}

class B{

public:
    void showFriendInfo(A &fa) {
        std::cout << fa.num << std::endl; // A 클래스의 private 변수에 접근 가능 
    }
}
```

### 함수의 friend 선언
* 전역 함수나 클래스 멤버 함수 대상으로도 friend 선언이 가능

```
class A{
public:
    B addB(const B&, const B&); 
}

class B{
private:
    int num;
public:
    friend B A::addB(const B&, const B&); // 멤버 함수 대상으로 friend 선언
    friend void showB(const B&); // 전역 함수 대상으로 friend 선언 
}

B A::addB(const B& b1, const B& b2){
    return B(b1.num + b2.num); // B 클래스의 private 변수에 접근 
}

void showB(const B& b){
    std::endl << b.num << std::endl; // B 클래스의 private 변수에 접근 
}
```

## static
* 전역 변수에 선언된 static: 선언된 파일 내에서만 참조를 허용. 
* 함수 내에 선언된 static: 한번만 초기화 되고, 지역변수와 달리 함수를 빠져나가도 소멸되지 않음.

### static 멤버 번수

```
class SimpleClass {
private:
    static int num; // static 멤버 변수 
}
int SimpleClass::num = 0; // 별도 초기화 

// sc1, sc2, sc3 모두에서 같은 num을 공유 
SimpleClass sc1;
SimpleClass sc2;
SimpleClass sc3;

```

* 클래스의 멤버 변수가 아님
* 초기화 별도 진행 (객체 생성과 동시에 생성되는 변수가 아니고, 이미 메모리에 할당이 이루어진 변수 이므로) 
* private 으로 선언되면 해당 클래스의 객체들만 접근 가능
* public 으로 선언되면 클래스의 이름을 통해서도 접근 가능 
* 선언된 클래스의 모든 객체가 동일한 static 변수 공유 

### static 멤버 함수

```
class SimpleClass {
private:
    int num1;
    static int num2; // static 멤버 변수 
public:
    static void addN(int n){ // static 멤버 함수 
        num1+=n; // (x)  함수는 클래스의 멤버가 아니므로 멤버 변수에 접근 불가능 
        num2+=n; // static 변수이므로 접근 가능 
    }
}

int SimpleClass::num2 = 0; // 별도 초기화 
```

* 선언된 클래스의 모든 객체가 동일한 static 함수 공유
* public으로 선언되면, 클래스의 이름을 이용해 호출 가능 
* 객체의 멤버로 존재하는 것이 아님
* static 멤버 함수 내에서는 static 으로 선언되지 않은 멤버 변수의 접근도, 멤버 함수의 호출도 불가능 하다


### const static 멤버
* 클래스 내에 선언된 const 멤버 변수의 초기화는 이니셜라이저를 통해서만 가능
* const static 변수는 바로 초기화 가능
> const static int var = 10;  

### mutable 키워드
* const 함수 내에서 값의 변경을 예외적으로 허용한다.

```
class SimpleClass{
private:
    int num1;
    mutable int num2;
public:
    SimpleClass(int n1, int n2):num1(n1), num2(n2) {}
    void copyToNum2() const { // const 함수 내에서는 값의 변경이 불가능 하지만 예외적으로 허용 
        num2 = num1; // 값 변경 가능 
    };
}

SimpleClass sc(20, 30);
sc.copyToNum2();

```

# 7. 상속의 이해

```cpp
class Parent{
private:
    int age;
public:
    Parent(int a, int n): age(a){}
    int getAge() {return age;}
};

class Children: public Parent {
private:
    int order;
public:
    Children(int a, int n, int o): Parent(a, n){
        order = o;
    }
    int getOrder() {return order};
};
```

* 자식 클래스의 생성자는 부모 클래스의 멤버까지 초기화할 의무가 있다 
* 부모 클래스의 멤버는 부모 클래스의 생성자를 호출하여 초기화 하는 것이 좋다
* 부모 클래스의 private 멤버 변수는 public getter, setter를 통해 접근 할 수 있다

## 자식 클래스의 생성 과정 
* 부모 클래스의 생성자는 100%로 호출된다
* 자식 클래스의 생성자에서 부모 클래스의 생성자 호출을 명시하지 않으면 부모 클래스의 default 생성자가 호출된다.
* 즉, 자식 클래스 생성자 부모 클래스 생성자 모두 호출 된다. 

## 자식 클래스의 소멸 과정
* 자식 클래스의 소멸자가 실행되고 난 다음에 부모 클래스의 소멸자가 실행된다. 

## Protected 선언
* private와 유사하게 외부에서는 접근이 불가능 하지만 클래스 내부에서는 접근 가능 
* 부모 클래스에서 private로 선언 했을 경우 자식 클래스에서 접근 가능 

## 세 가지 형태의 상속
```cpp
class Parent{
private:
    int num1;
protected:
    int num2;
public:
    int num3;
};

```

* protected 상속: protected 보다 접근 범위가 넓은 멤버는 protected로 변경시켜 상속한다
```cpp
class Children : protected Parent{
private:
    int num1;
protected:
    int num2;
protected:
    int num3;
};
```

* private 상속: private 보다 접근 범위가 넓은 멤버는 private로 변경시켜 상속한다.

```cpp
class Children : protected Parent{
private:
    int num1;
private:
    int num2;
private:
    int num3;
};
```

* public 상속: private을 제외한 나머지는 그냥 그대로 상속한다.

## 상속을 위한 조건
* Is-A 관계의 성립: 자식 클래스는 부모 클래스가 가진 모든 것을 지니고, 거기에 자식 클래스 만의 특성이 더해짐 
  * 무선 전화기 is-a 전화기
  * 노트북 컴퓨터 is-a 컴퓨터
  * 고객 is-a 사람
  * 강아지 is-a 동물

* Has-A 관계는 보통 클래스의 멤버 변수로 표현 
  * 경찰 has-a 총
  
# 8. 상속과 다형성
## 객체 포인터의 참조 관계
* 객체 포인터 변수: 객체의 주소 값을 저장하는 포인터 변수
  *  부모 클래스를 상속하는 자식 클래스의 객체도 가리킬 수 있다. 
  > Parent * ptr = new Children();
  
  * 즉, 클래스 포인터 변수는 해당 클래스 또는 해당 클래스를 직접 혹은 간접적으로 상속하는 모든 객체를 가리킬 수 있다.
  
* 자식 클래스의 함수가 부모 클래스와 둥일한 이름과 형태로 함수를 정의할 경우 함수 오버라이딩이 이루어진다.
  * 자식 클래싀 함수에 의해 부모 클래스 함수는 가려진다. 
  * 부모 클래스 명시를 통해 부모 클래스 함수를 호출할 수 있다. 
  
```cpp
Children c;
c.func(); // 자식 클래스 함수 오버라이딩 
c.Parent::func(); // 부모 클래스 함수 
```

## Virtual function
* C++ 컴파일러는 포인터 연산의 가능성 여부 판단시, __포인터의 자료형을 기준__ 으로 판단하지, 실제 가리키는 객체의 자료형을 기준으로 판단하지 않는다.
```cpp
Parent * ptr = new Children(); // 컴파일 OK
ptr->ChildrenFunc(); // 컴파일 error
ptr->ParentFunc(); // 컴파일 OK 
Children * cPtr = ptr; // 컴파일 error
```

* ptr이 실제 가리키는 객체가 Children이었다는 사실을 기억하지 않아 컴파일 에러
* 포인터 형에 해당하는 클래스에 정의된 멤버만 접근이 가능하다. 

```cpp

class First{
public:
    void show();
};

class Second : public First{
public:
    void show();
};

First * ptr = new Second();
ptr->show(); // First 클래스의 함수가 호출 됨 
```

* 함수가 오버라이딩 됬을 경우, 각 포인터형의 class에 선언된 함수를 호출한다

```cpp
Children * cPtr = new Children();
Parent * ptr = cPtr; // 컴파일 OK
```

* 해당 클래스를 직간접 적으로 상속하는 클래스를 모두 가리킬 수 있으므로 컴파일 성공 

### Virtual Func
* 함수를 오버라이딩 하는 것은, 해당 객체에서 호출되어야 하는 함수를 바꾼다는 의미인데, 포인터 변수에 따라 호출 함수 종류가 달라지는 것은 문제거 있음
* virtual로 가상 함수 선언을 할 경우 포인터 자료형을 기반으로 호출 대상을 결정하지 않고, 포인터 변수가 가리키는 실제 객체를 참조하여 호출의 대상을 결정한다.

```cpp

class First{
public:
    virtual void show();
};

class Second : public First{
public:
    virtual void show();
};

First * ptr = new Second();
ptr->show(); // Second 클래스의 함수가 호출 됨 
```

### 순수 가상 함수와 추상 클래스 
* 클래스 중에는 객체 생성을 목적으로 정의되지 않은 클래스도 존재한다. 
* 함수의 몸체가 정의되지 않은 함수 (순수 가상 함수)를 선언하여 이런 객체 생성을 문법적으로 막는 것이 좋다.
    * 함수 = 0 형식으로 표시 
* 이런 순수 가상 함수를 포함하는 클래스를 추상 클래스라고 한다. 
* 부모 클래스에 virtual 선언을 했으면 자식 class에는 별도로 virtual 표시를 하지 않아도 된다. 

```cpp
class Parent{ // 추상 클래스 
public:
    Parent(){ ... }
    virtual getInfo() = 0; // 순수 가상 함수 
    virtual setNum(int num) = 0; // 순수 가상 함수 
};
```

### 가상 소멸자
* 포인터 변수의 자료형에 상관 없이 모든 소멸자가 호출되어야 한다
* 그렇기 위해서 소멸자에 Virtual 선언을 추가해야 한다.

### 참조자
* 해당 클래스의 참조자는 해당 객체 또는 해당 클래스를 직간접 적으로 상속하는 모든 객체를 참조할 수 있다. 
* 포인터와 마찬가지로 참조자의 자료형에 해당하는 멤버 함수만 호출 가능 
* virtual 사용시 참조자가 가리키는 객체의 멤버 함수 호출 가능 

# 9. Virtual 의 원리와 다중 상속 

## 멤버 함수와 가상함수의 동작 원리 
* 객체 안에 맴버 함수가 존재 하는가?
  * 객체가 생성되면 멤버 변수는 객체 내에 존재하지만, 멤버 함수는 메모리 공간에 별도로 위치하고, 이 함수가 정의된 클래스의 모든 객체가 이를 공유하는 형태
* 한 개 이상 가상 함수를 포함하는 클래스는 컴파일러가 Virtual Table을 만든다.  

```cpp
class AAA {
public:
  virtual void Func1();
  virtual void Func2();
}
class BBB : public AAA {
public:
  virtual void Func1();
  void Func3();
}
```

< class AAA V-Table>

| key | value |
|---|---|
|void AAA::Func1() | 0x1024 번지 |
|void AAA::Func2() | 0x2048 번지 |

< class BBB V-Table>

| key | value |
|---|---|
|void BBB::Func1() | 0x3072 번지 |
|void AAA::Func2() | 0x2048 번지 |
|void BBB::Func3() | 0x4096 번지 |

오버라이딩 된 가상함수 Func1() 에 대한 정보가 존재 하지 않는다. 그래서 포인터 변수가 가리키는 실제 객체의 함수를 호출 할 수 있다.

* 가상 함수가 포함되면, 가상함수 테이블이 생성되고, 이 테이블을 참조하여 호출될 함수가 결정되기 때문에, 실행 속도가 감소하지만, 극히 미미하여 유용하게 활용된다. 

## 다중 상속에 대한 이해 
* 다중 상속의 모호성
  * 두 기초 클래스에 동일한 이름의 멤버가 존재하는 경우 
  

```cpp
class One {
public:
  void func();
};

class Two {
public:
  void func();
};

class MultiDerived: public One, protected Two {
public:
  void ComplexFunc(){
    One::func(); // 범위 지정 연산자 사용 
    Two::func();
  }
};
```

* 가상 상속 
  * 같은 클래스를 다중 상속하게 될 경우, 가상 상속 선언 시, 생성자가 한번 만 호출된다.
  

```cpp
class Base { ... };

class MiddleDerivedOne: virtual public Base { ... }; // 가상 상속

class MiddleDerivedTwo: virtual public Base { ... }; // 가상 상속  

class LastDerived: public MiddleDerivedOne,  public MiddleDerivedTwo { ... }; // Base 생성자가 한번만 호출 
```

# 10. 연산자 오버로딩 1

* 연산자를 오버로딩 하는 두 가지 방법
  * 멤버함수에 의한 연산자 오버로딩
  * 전역함수에 의한 연산자 오버로딩
  

```cpp
class Point{
private:
  int x;
  int y;
public:
  Point(int _x, int _y): x(_x), y(_y) {}
  
  // 멤버 함수  pos1.operator+(pos2)
  Point operator+(const Point& ref){
    Point p(ref.x + x, ref.y + y);
    return p;
  } 
}

// 전역 함수 operator+(pos1, pos2)
Point operator+(const Point& pos1, const Point& pos2){
  Point p(pos1.x + pos2.x, pos1.y + pos2.y);
  return p;
}
```

* 멤버 함수로만 오버로딩 가능한 연산자
> =, (), [], ->  

* 연산자가 오버로딩 되어도 우선순위와 결합성은 바뀌지 않는다.
* 연산자 오버로딩 함수는 매개 변수의 디폴트값 설정이 불가능하다. 
* 연산자의 기본 기능을 변경하는 형태의 오버로딩은 허용되지 않는다.

## 단항 연산자의 오버로딩
* 1 증가 연산자: ++
* 1 감소 연산자: --

```cpp
class Point{
private:
  int x;
  int y;
public:
  Point(int _x, int _y): x(_x), y(_y) {}
  
  // 전위 증가 ++pos
  Point& operator++(){ 
    x+=1;
    y+=1;
    return *this;
  }
  // 후위 증가 pos++
  const Point operator++(int) {
    const Point pos(x, y); // 값의 변경을 허용하지 않겠다 
    x+=1;
    y+=1;
    return pos;
  }
  
  friend Point& operator++(Point &ref); // private 변수 접근을 허용하기 위한 선언 
  friend const Point operator++(Point &ref, int);
}

// 전위 증가 ++pos
Point& operator++(Point &ref){
  ref.x+=1;
  ref.y+=1;
  return ref;
}

// 후위 증가 pos++
const Point operator++(Point &ref, int){
  const Point pos(ref.x, ref.y);
  ref.x+=1;
  ref.y+=1;
  return pos;
}

Point pos(3, 6);
(pos++)++;  // 컴파일 에러 (const 임시객체가 생성되고 이를 변경하려고 하기 때문에)  
++(++pos)l // 컴파일 OK 

```

## 교환 법칙 문제의 해결 

```cpp
class Point{
private:
  int x;
  int y;
public:
  Point(int _x, int _y): x(_x), y(_y) {}
  
  Point operator*(int times){
    Point pos(x*times, y*times);
    return pos;
  }
}

Point pos(1, 2);
Point pos2 = pos*3; // 컴파일 성공  
Point pos3 = 3*pos; // 컴파일 에러 
```

* 교환법칙이 성립되게 구현하려면 정역함수 형태로 오버로딩 하는 수 밖에 없다

```cpp
Point operator* (int times, Point & ref){
  return ref*times; // 멤버 함수 호출 
}
```

# 11. 연산자 오버로딩 2

## 반드시 해야하는 대입 연산자의 오버로딩

* 정의 하지 않으면 디폴트 대입 연산자가 삽입된다. 
* 다폴트 대입 연산자는 멤버 대 멤버의 복사(얕은 복사)를 진행한다.
* 연산자 내에서 동적 할당을 한다면, 그리고 깊은 복사가 필요하다면 직접 정의해야 한다. 

* 자식 클래스의 대입 연산자 정의에서 명시적으로 부모 클래스의 대입 연산자를 호출하지 않으면, 부모 클래스의 대입 연산자는 호출되지 않아서 멤버 대 멤버의 복사 대상에서 제외된다.

## 배열의 인덱스 연산자 오버로딩

* C, C++은 배열의 경게검사를 하지 않는다. 
* operator[ ]

```cpp
#include <cstdlib>

class BoundCheckIntArray{
private:
  int * arr;
  int arrlen;
  
  BoundCheckIntArray(const BoundCheckIntArray& ref) { }
  BoundCheckIntArray& operator=(const BoundCheckIntArray& arr) { } // 데이터의 유일성 보장을 위해 복사와 대입을 막음 
  
public:
  BoundCheckIntArray(int len): arrlen(len) {
    arr = new int[len];
  }
  
  int& operator[] (int idx){
    if (idx < 0 || inx >= arrlen){
      std::cout << "Out of bound " << std::endl;
      exit(1);
    }
    return arr[idx];
  }
  
  // const 를 이용한 오버로딩, 오직 const 함수 내에서만 호출 가능 
  int operator[] (int idx) const {
      if (idx < 0 || inx >= arrlen){
      std::cout << "Out of bound " << std::endl;
      exit(1);
    }
    return arr[idx];
  }
  
  ~BoundCheckIntArray() {
    delete []arr;
  }
}
```

* 스마트 포인터
  * 감싼 객체의 소멸 연산이 자동으로 이루어짐 
  * ->, * 연산자 오버로딩으로 실제 포인터가 가리키는 객체 제어 가능 

```cpp
class Point {
private:
  int x;
  int y;
public:
  Point(int _x, int _y): x(_x), y(_y) {}
  void SetPos(int x, int y) {
    this->x = x;
    this->y = y;
  }
}

class SmartPtr{
private:
  Point * ptr;
public:
  SmartPtr(Point * _ptr): ptr(_ptr) { }
  Point& operator*() const { // 포인터가 가리키는 객체 접근 
    return *ptr;
  }
  Point* operator->() const{ // 포인터가 가리키는 객체 멤버 접근
    return ptr;
  }
  ~SmartPtr() {
    delete ptr;
  }
}

SmartPtr sptr(new Point(1, 2));
sptr->SetPos(10, 20); // -> 오버로딩으로 멤버 함수에 접근 가능 

```

# 12. String 클래스의 디자인
* \<string\> 헤더파일 사용

# 13. 템플릿(Template) 1

## 템플릿에 대한 이해와 함수 템플릿
* template은 모형자를 만들어 내는 것
* 함수 템플릿

```cpp
template <typename T>
T add(T num1, T num2){
  return num1+num2;
}
```

* typename 대신 class를 사용해도 된다 
> template <class T> 

* 컴파일 할 때 자료형 별 함수가 하나씩 만들어 진다. 
  * 전달 되는 인자의 자료형을 참조하여 호출된 함수의 유형을 컴파일러가 결정 
  * 이를 템플릿 함수라고 한다. 
* 호출 시, 자료형 정보를 명시해도 되고, 생략해도 된다. 
  
```cpp
int add<int> (int num1, int num2){
  return num1+num2;
}

double add<double> (double num1, double num2){
  return num1+num2;
}

int num1 = add(10, 20);
double num2 = add<double>(10.2, 20.5);
```

* 둘 이상의 타입에 대해서 함수 템플릿 선언

```cpp
template <typename T1, typename T2>
void ShowData(double num){
  std::cout << (T1) num << (T2) num << std::endl;
}
```

* 템플릿 한수의 특수화
  * 직접 제시를 할테니, 해당 타입의 템플릿 함수가 필요한 경우에 별도로 만들지 말고 이것을 써라
  
```cpp
#include <cstring>

template <>
char* Max(char* a, char* b){
  return strlen(a) > strlen(b) ? a: b;
}
```

## 클래스 템플릿 (Class Template)

```cpp
// Point.h 파일

template <typename T>
class Point{
private:
  T x;
  T y;
public:
  Point(T _x, T_y): x(_x), y(_y) { }
  void add(T _x, T _y) {}
}

Point<int> pos(10, 20); // 호출 시 자료형 생략 불가능  
```

* 클래스 템플릿 기반의 객체 생성에는 반드시 자료형을 명시하도록 되어 있다
* 템플릿의 멤버 함수를 클래스 외부에서 정의하는 것이 가능하다.


```cpp
// Point.cpp 파일 

template <typename T>
void Point<T>::add(T _x, T _y){
  x+=_x;
  y+=_y;
}
```

```cpp
// main.cpp 파일 

#include "Point.h"
#include "Point.cpp" // 컴파일러에게 모든 정보를 알려줘야함 

int main(){
  Point<int> pos(3, 4);
  Point<double> pos(3.4, 4.5);
  Point<char> pos('a', 'b');
}
```

* 컴파일은 파일단위로 이뤄진다. main.cpp 파일을 컴파일 할 때 컴파일러는 총 3개의 템플릿 클래스를 생성해야 한다.
따라서 클래스 템플릿 Point 의 모든 것을 알아야 하기 때문에 "Point.cpp" 파일도 함께 include 해야 한다. 
  
## 타입이 아닌 템플릿 인자 

```cpp
template <typename T, int num>
T add_num(T t){
    return t+num;
}

int main(){
    int x = 3;
    std::count << "x: " << add_num<int, 5>(x) << std::endl; // x: 8
}
```

* 템플릿 인자로 전달할 수 있는 타입: bool, char, int, long (double, float 제외), 포인터 타입, enum, 널포인터 

### 디폴트 템플릿 인자 

```cpp
template <typename T, int num=5>
T add_num(T t){
    return t+num;
}
```
  
## 가변 길이 템플릿
* 임의의 개수 인자를 받는 템플릿을 작성할 수 있다. 

```cpp
#include <iostream>

template <typename T>
void print(T arg) {
std::cout << arg << std::endl;
}

template<typename T, typename... Types>
void print(T arg, Types... args) {
    std::cout << arg << ", ";
    print(args...);
}

int sum_all() { return 0; }

template <typename... Ints>
int sum_all(int num, Ints... nums) {
return num + sum_all(nums...);
}

```

* 순서도 유의해야 한다. c++ 컴파일러는 함수를 컴파일 시, 자신의 앞에 정의되어 있는 함수들 밖에 보지 못하기 때문에 break 케이스 함수가 먼저 정의되어 있어야 한다. 
* ... : 템플릿 파라미터 팩, 0개 이상의 템플릿 인자들을 나타냄 
* 재귀함수 형태로 만들어야 하므로, 반드시 종료를 위한 베이스 케이스를 따로 정의해야 한다. 

## Fold Expression

```cpp
template<typename... Ints>
int sum_all(Ints... nums){
    return (...+nums);
}

class A {
public:
  void do_something(int x) const {
    std::cout << "Do something with " << x << std::endl;
  }
};

template <typename T, typename... Ints>
void do_many_things(const T& t, Ints... nums) {
    // 각각의 인자들에 대해 do_something 함수들을 호출한다.
    (t.do_something(nums), ...);
}
int main() {
  A a;
  do_many_things(a, 1, 3, 2, 4);
}
```

* c++17에 새로 도입된 형식으로 , 재귀함수 형태처럼 베이스 케이스가 필요 없다 

## 템플릿 메타 프로그래밍 (TMP)
* 타입은 반드시 컴파일 타임에 확정되어야 하므로, 컴파일 타임에 모든 연산이 끝난다.
* 이렇게 컴파일 타임에 생성되는 코드로 프로그래밍을 하는 것을 메타 프로그래밍이라고 한다. 

```cpp
template<int N>
struct Factorial {
    static const int result = N * Factorial<N-1>::result;
};

template<>
struct Factorial<1> {
    static const int result = 1;
};

int main() {
    int result = Factorial<6>::result;
}
```

* 장점: 어떠한 코드든, TMP로 변환할 수 있다. 모두 컴파일 타임에 모든 연산이 끝나기 때문에 프로그램의 실행속도를 향상시킬 수 있다. (컴파일 시간은 늘어남)
* 단점: 버그를 찾는것이 매우 어려움, 디버깅 불가능, 오류의 길이가 매우 김 

## auto
* 컴파일 시 컴파일러에 의해 타입이 추론 됨 



