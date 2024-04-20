#ifndef SMARTPTRBUILDER_H_
#define SMARTPTRBUILDER_H_

#include "EnableSharedFromThis.h"
#include <memory>

namespace Generic
{

class SmartPtrBuilder
{
public:
  template< typename T0, typename T >
  static std::shared_ptr< T > CreateSharedPtr(T *iPtr);

  template< typename T0, typename T1, typename T >
  static std::shared_ptr< T > CreateSharedPtr(T *iPtr);

  template< typename T0, typename T1, typename T2, typename T >
  static std::shared_ptr< T > CreateSharedPtr(T *iPtr);

  template< typename T0, typename T1, typename T2, typename T3, typename T >
  static std::shared_ptr< T > CreateSharedPtr(T *iPtr);

  template< typename T0, typename T1, typename T2, typename T3, typename T4, typename T >
  static std::shared_ptr< T > CreateSharedPtr(T *iPtr);

  template< typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T >
  static std::shared_ptr< T > CreateSharedPtr(T *iPtr);

  template< typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T >
  static std::shared_ptr< T > CreateSharedPtr(T *iPtr);

  template< typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T >
  static std::shared_ptr< T > CreateSharedPtr(T *iPtr);

  template< typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T >
  static std::shared_ptr< T > CreateSharedPtr(T *iPtr);

  template< typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T >
  static std::shared_ptr< T > CreateSharedPtr(T *iPtr);

  template< typename T0, typename T >
  static void AssignWeakPtr(std::shared_ptr< T > &ioSharedPtr);

  template< typename T0, typename T1, typename T >
  static void AssignWeakPtr(std::shared_ptr< T > &ioSharedPtr);

  template< typename T0, typename T1, typename T2, typename T >
  static void AssignWeakPtr(std::shared_ptr< T > &ioSharedPtr);

  template< typename T0, typename T1, typename T2, typename T3, typename T >
  static void AssignWeakPtr(std::shared_ptr< T > &ioSharedPtr);

  template< typename T0, typename T1, typename T2, typename T3, typename T4, typename T >
  static void AssignWeakPtr(std::shared_ptr< T > &ioSharedPtr);

  template< typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T >
  static void AssignWeakPtr(std::shared_ptr< T > &ioSharedPtr);

  template< typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T >
  static void AssignWeakPtr(std::shared_ptr< T > &ioSharedPtr);

  template< typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T >
  static void AssignWeakPtr(std::shared_ptr< T > &ioSharedPtr);

  template< typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T >
  static void AssignWeakPtr(std::shared_ptr< T > &ioSharedPtr);

  template< typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T >
  static void AssignWeakPtr(std::shared_ptr< T > &ioSharedPtr);

private:
  SmartPtrBuilder();
};

template< typename T0, typename T >
inline std::shared_ptr< T > SmartPtrBuilder::CreateSharedPtr(T *iPtr)
{
  std::shared_ptr< T > wSharedPtr(iPtr);
  AssignWeakPtr< T0 >(wSharedPtr);
  return wSharedPtr;
}

template< typename T0, typename T1, typename T >
inline std::shared_ptr< T > SmartPtrBuilder::CreateSharedPtr(T *iPtr)
{
  std::shared_ptr< T > wSharedPtr(iPtr);
  AssignWeakPtr< T0, T1 >(wSharedPtr);
  return wSharedPtr;
}

template< typename T0, typename T1, typename T2, typename T >
inline std::shared_ptr< T > SmartPtrBuilder::CreateSharedPtr(T *iPtr)
{
  std::shared_ptr< T > wSharedPtr(iPtr);
  AssignWeakPtr< T0, T1, T2 >(wSharedPtr);
  return wSharedPtr;
}

template< typename T0, typename T1, typename T2, typename T3, typename T >
inline std::shared_ptr< T > SmartPtrBuilder::CreateSharedPtr(T *iPtr)
{
  std::shared_ptr< T > wSharedPtr(iPtr);
  AssignWeakPtr< T0, T1, T2, T3 >(wSharedPtr);
  return wSharedPtr;
}

template< typename T0, typename T1, typename T2, typename T3, typename T4, typename T >
inline std::shared_ptr< T > SmartPtrBuilder::CreateSharedPtr(T *iPtr)
{
  std::shared_ptr< T > wSharedPtr(iPtr);
  AssignWeakPtr< T0, T1, T2, T3, T4 >(wSharedPtr);
  return wSharedPtr;
}

template< typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T >
inline std::shared_ptr< T > SmartPtrBuilder::CreateSharedPtr(T *iPtr)
{
  std::shared_ptr< T > wSharedPtr(iPtr);
  AssignWeakPtr< T0, T1, T2, T3, T4, T5 >(wSharedPtr);
  return wSharedPtr;
}

template< typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T >
inline std::shared_ptr< T > SmartPtrBuilder::CreateSharedPtr(T *iPtr)
{
  std::shared_ptr< T > wSharedPtr(iPtr);
  AssignWeakPtr< T0, T1, T2, T3, T4, T5, T6 >(wSharedPtr);
  return wSharedPtr;
}

template< typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T >
inline std::shared_ptr< T > SmartPtrBuilder::CreateSharedPtr(T *iPtr)
{
  std::shared_ptr< T > wSharedPtr(iPtr);
  AssignWeakPtr< T0, T1, T2, T3, T4, T5, T6, T7 >(wSharedPtr);
  return wSharedPtr;
}

template< typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T >
inline std::shared_ptr< T > SmartPtrBuilder::CreateSharedPtr(T *iPtr)
{
  std::shared_ptr< T > wSharedPtr(iPtr);
  AssignWeakPtr< T0, T1, T2, T3, T4, T5, T6, T7, T8 >(wSharedPtr);
  return wSharedPtr;
}

template< typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T >
inline std::shared_ptr< T > SmartPtrBuilder::CreateSharedPtr(T *iPtr)
{
  std::shared_ptr< T > wSharedPtr(iPtr);
  AssignWeakPtr< T0, T1, T2, T3, T4, T5, T6, T7, T8, T9 >(wSharedPtr);
  return wSharedPtr;
}

template< typename T0, typename T >
inline void SmartPtrBuilder::AssignWeakPtr(std::shared_ptr< T > &ioSharedPtr)
{
  ioSharedPtr->EnableSharedFromThis< T0 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
}

template< typename T0, typename T1, typename T >
inline void SmartPtrBuilder::AssignWeakPtr(std::shared_ptr< T > &ioSharedPtr)
{
  ioSharedPtr->EnableSharedFromThis< T0 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T1 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
}

template< typename T0, typename T1, typename T2, typename T >
inline void SmartPtrBuilder::AssignWeakPtr(std::shared_ptr< T > &ioSharedPtr)
{
  ioSharedPtr->EnableSharedFromThis< T0 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T1 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T2 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
}

template< typename T0, typename T1, typename T2, typename T3, typename T >
inline void SmartPtrBuilder::AssignWeakPtr(std::shared_ptr< T > &ioSharedPtr)
{
  ioSharedPtr->EnableSharedFromThis< T0 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T1 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T2 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T3 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
}

template< typename T0, typename T1, typename T2, typename T3, typename T4, typename T >
inline void SmartPtrBuilder::AssignWeakPtr(std::shared_ptr< T > &ioSharedPtr)
{
  ioSharedPtr->EnableSharedFromThis< T0 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T1 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T2 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T3 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T4 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
}

template< typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T >
inline void SmartPtrBuilder::AssignWeakPtr(std::shared_ptr< T > &ioSharedPtr)
{
  ioSharedPtr->EnableSharedFromThis< T0 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T1 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T2 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T3 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T4 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T5 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
}

template< typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T >
inline void SmartPtrBuilder::AssignWeakPtr(std::shared_ptr< T > &ioSharedPtr)
{
  ioSharedPtr->EnableSharedFromThis< T0 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T1 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T2 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T3 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T4 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T5 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T6 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
}

template< typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T >
inline void SmartPtrBuilder::AssignWeakPtr(std::shared_ptr< T > &ioSharedPtr)
{
  ioSharedPtr->EnableSharedFromThis< T0 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T1 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T2 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T3 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T4 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T5 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T6 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T7 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
}

template< typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T >
inline void SmartPtrBuilder::AssignWeakPtr(std::shared_ptr< T > &ioSharedPtr)
{
  ioSharedPtr->EnableSharedFromThis< T0 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T1 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T2 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T3 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T4 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T5 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T6 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T7 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T8 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
}

template< typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T >
inline void SmartPtrBuilder::AssignWeakPtr(std::shared_ptr< T > &ioSharedPtr)
{
  ioSharedPtr->EnableSharedFromThis< T0 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T1 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T2 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T3 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T4 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T5 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T6 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T7 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
  ioSharedPtr->EnableSharedFromThis< T9 >::SetWeakPtr(ioSharedPtr, ioSharedPtr.get());
}

}

#endif /* SMARTPTRBUILDER_H_ */

// Copyright (C) 2011 by Philippe Cayouette

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
