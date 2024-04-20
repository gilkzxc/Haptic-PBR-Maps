#ifndef ENABLESHAREDFROMTHIS_H_
#define ENABLESHAREDFROMTHIS_H_

#include <memory>

namespace Generic
{

template< typename T >
class EnableSharedFromThis
{
public:
  std::shared_ptr< T > SharedFromThis();
  std::shared_ptr< T const > SharedFromThis() const;

  virtual ~EnableSharedFromThis();
protected:
  EnableSharedFromThis();
  EnableSharedFromThis(EnableSharedFromThis const &);
  EnableSharedFromThis & operator=(EnableSharedFromThis const &);

private:
  friend class SmartPtrBuilder;

  template< typename X, typename Y >
  void SetWeakPtr(std::shared_ptr< X > const &iSharedThis, Y *iThis) const;

  mutable std::weak_ptr< T > mWeakThis;
};

template< typename T >
EnableSharedFromThis< T >::EnableSharedFromThis()
{
}

template< typename T >
EnableSharedFromThis< T >::EnableSharedFromThis(EnableSharedFromThis const &)
{
}

template< typename T >
EnableSharedFromThis< T >::~EnableSharedFromThis()
{
}

template< typename T >
EnableSharedFromThis< T > &EnableSharedFromThis< T >::operator=(EnableSharedFromThis< T > const &)
{
  return *this;
}

template< typename T >
std::shared_ptr< T > EnableSharedFromThis< T >::SharedFromThis()
{
  std::shared_ptr< T > wSharedThis(mWeakThis);
  return wSharedThis;
}

template< typename T >
std::shared_ptr< T const > EnableSharedFromThis< T >::SharedFromThis() const
{
  std::shared_ptr< T const > wSharedThis(mWeakThis);
  return wSharedThis;
}

template< typename T >
template< typename X, typename Y >
void EnableSharedFromThis< T >::SetWeakPtr(std::shared_ptr< X > const &iSharedThis, Y *iThis) const
{
  if (mWeakThis.expired())
  {
    mWeakThis = std::shared_ptr< T >(iSharedThis, iThis);
  }
}

}

#endif /* ENABLESHAREDFROMTHIS_H_ */

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
