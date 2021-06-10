---
layout: post  
title: (컴퓨터공학과 전공 기초 시리즈) C++ 프로그래밍         
subtitle: Programming     
tags: [c++, cpp, programing language, computer science, basic knowledge]    
comments: true  
---  
기본적으로 알고 있어야하는 __*Basic*__ 한 내용만을 다룹니다.  
윤성우 열혈 c++ 프로그래밍을 참조하여 작성하였습니다.

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

typedef _Car2 {
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
    * 호출 시 .(dot) 을 통해서 함수 접근 가능

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
* 클래스 기반의 두가지 객체 생성 방법 
```cpp
Car basicCar; // 일반적인 변수의 선언 방식 
Car *basicCarPtr = new Car; // 동적 할당 방식 
```
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
// SimpleClass sc(); (x)
// SimpleClass sc; (o)

SimpleClass *scPtr = new SimpleClass(10); 
// SimpleClass *scPtr = new SimpleClass(); (o)
// SimpleClass *scPtr = new SimpleClass; (o)
```
* 멤버 이니셜라이저를 이용한 초기화
    * 선언과 동시에 초기화가 이루어짐
    * const 변수도 초기화 가능 (const 변수는 선언과 동시에 초기화 되어야함)
    * 참조자도 초기화 가능 (참조자는 선언과 동시에 초기화 되어야함)
```cpp
class SimpleClass{
private:
    int num1;
    const int num2;
    int &num3;
    
public:
    SimpleClass(int n1, int n2, int n3): num1(n1), num2(n2), num3(n3) {}
};
```
* 디폴트 생성자
    * 생성자를 정의 하지 않았을때만 디폴트 생성자가 생성됨 

### 소멸자
* 객체 소멸시 반드시 호출
* 클래스의 이름앞에 ~가 붙은 형태의 이름을 갖음
* 반환형이 선언되지 않으ㅁ, 반환하지 않음
* 매개변수는 void 형으로 선언되어야 하므로, 오버로딩이 불가능하다.

```cpp
class SimpleClass{
private:
    char* name;
public:

    SimpleClass() {}

    SimpleClass(int len){
        name = new char[len+1];
    }
    ~SimpleClass(){
        delete []name;
    }
}
```
## 클래스 배열 
### 객체 배열
> SimpleClass arr[100];   
> SimpleClass * arrPtr = new SimpleClass[100];
* 배열을 선언하는 경우에도 생성자가 호출이 됨
* 단 인자를 전달하지 못하므로 디폴트 생성자 형태의 생성자가 정의되어 있어야 한다.
* 배열이 소멸할 때, 객체 소멸자가 호출 됨

### 객체 포인터 배열
> SimpleClass * arrPtr[100];
```cpp
int len = 10;
SimpleClass * arrPtr[len];

for (int i=0; i<len; i++){
    arrPtr[i] = new SimpleClass();
    arrPtr[i]->printName(); // 객체 포인터에서 멤버 접근은 -> 를 사용 
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