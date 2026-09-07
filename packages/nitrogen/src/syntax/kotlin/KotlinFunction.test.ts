import assert from 'node:assert/strict'
import { after, before, describe, it } from 'node:test'
import { NitroConfig } from '../../config/NitroConfig.js'
import { BooleanType } from '../types/BooleanType.js'
import { FunctionType } from '../types/FunctionType.js'
import { Int64Type } from '../types/Int64Type.js'
import { NamedWrappingType } from '../types/NamedWrappingType.js'
import { NumberType } from '../types/NumberType.js'
import { OptionalType } from '../types/OptionalType.js'
import { StringType } from '../types/StringType.js'
import type { Type } from '../types/Type.js'
import { UInt64Type } from '../types/UInt64Type.js'
import { VoidType } from '../types/VoidType.js'
import { getJNINativeRegistrations } from './JNINativeRegistrations.js'
import { KotlinCxxBridgedType } from './KotlinCxxBridgedType.js'
import { createKotlinFunction } from './KotlinFunction.js'

function generateCallback(
  returnType: Type,
  parameterType?: Type,
  language: 'c++' | 'kotlin' = 'c++'
): string {
  const parameters =
    parameterType == null ? [] : [new NamedWrappingType('value', parameterType)]
  const callback = new FunctionType(
    returnType,
    parameters,
    returnType.kind !== 'void'
  )
  const file = createKotlinFunction(callback).find(
    (f) => f.language === language
  )
  assert.ok(file)
  return file.content
}

describe('Kotlin callback JNI returns', () => {
  const originalConfig = Object.getOwnPropertyDescriptor(
    NitroConfig,
    'current'
  )!
  const registrations = getJNINativeRegistrations()
  const registrationCount = registrations.length

  before(() => {
    const config = new NitroConfig({
      cxxNamespace: ['test'],
      ios: { iosModuleName: 'NitroTest' },
      android: {
        androidNamespace: ['test'],
        androidCxxLibName: 'NitroTest',
      },
      autolinking: {},
      gitAttributesGeneratedFlag: false,
    })
    Object.defineProperty(NitroConfig, 'current', { get: () => config })
  })

  after(() => {
    Object.defineProperty(NitroConfig, 'current', originalConfig)
    registrations.length = registrationCount
  })

  // FunctionN-derived Kotlin interfaces expose boxed primitive returns, but
  // specialize primitive parameters. These signatures were checked with kotlinc.
  for (const [name, type, primitive, boxed, result] of [
    ['Double', new NumberType(), 'double', 'JDouble', '__result->value()'],
    [
      'Boolean',
      new BooleanType(),
      'jboolean',
      'JBoolean',
      'static_cast<bool>(__result->value())',
    ],
    ['Long', new Int64Type(), 'int64_t', 'JLong', '__result->value()'],
  ] as const) {
    it(`unboxes ${name} callback returns while preserving primitive parameters and invoke_cxx`, () => {
      const code = generateCallback(type, type)
      assert.ok(
        code.includes(
          `getMethod<jni::local_ref<jni::${boxed}>(${primitive} /* value */)>("invoke")`
        )
      )
      assert.ok(code.includes(`return ${result};`))
      assert.ok(code.includes(`${primitive} invoke_cxx(${primitive} value)`))
    })
  }

  it('keeps Unit callbacks void even when the caller requests boxing', () => {
    const type = new VoidType()
    const bridge = new KotlinCxxBridgedType(type)
    assert.equal(bridge.asJniReferenceType('local', true), 'void')
    const code = generateCallback(type, new Int64Type())
    assert.ok(code.includes('getMethod<void(int64_t /* value */)>("invoke")'))
    assert.ok(code.includes('method(self(), value);'))
    assert.ok(code.includes('void invoke_cxx(int64_t value)'))
    assert.ok(!code.includes('jni::local_ref<void>'))
  })

  it('adapts ULong callbacks to stable JNI methods while preserving the Kotlin type', () => {
    const type = new UInt64Type()
    const bridge = new KotlinCxxBridgedType(type)
    assert.equal(bridge.asJniReferenceType('alias'), 'jlong')
    for (const reference of ['alias', 'local', 'global'] as const) {
      assert.equal(
        bridge.asJniReferenceType(reference, true),
        `jni::${reference}_ref<jni::JLong>`
      )
    }
    const code = generateCallback(type)
    assert.ok(
      code.includes('getMethod<jni::local_ref<jni::JLong>()>("invoke_jni")')
    )
    assert.ok(code.includes('return static_cast<uint64_t>(__result->value());'))
    assert.ok(code.includes('jlong invoke_cxx()'))
    const kotlin = generateCallback(type, type, 'kotlin')
    assert.ok(kotlin.includes('override fun invoke(value: ULong): ULong'))
    assert.ok(kotlin.includes('fun invoke_jni(value: Long): Long?'))
    assert.ok(kotlin.includes('return invoke(value.toULong()).toLong()'))
    assert.ok(kotlin.includes('= invoke_cxx(value.toLong()).toULong()'))
    assert.ok(
      kotlin.includes('private external fun invoke_cxx(value: Long): Long')
    )
  })

  it('adapts nullable ULong callback values without losing null', () => {
    const type = new OptionalType(new UInt64Type())
    const kotlin = generateCallback(type, type, 'kotlin')
    assert.ok(kotlin.includes('override fun invoke(value: ULong?): ULong?'))
    assert.ok(kotlin.includes('fun invoke_jni(value: Long?): Long?'))
    assert.ok(
      kotlin.includes(
        'return invoke(value?.let { it.toULong() })?.let { it.toLong() }'
      )
    )
    assert.ok(
      kotlin.includes(
        '= invoke_cxx(value?.let { it.toLong() })?.let { it.toULong() }'
      )
    )
  })

  it('keeps Unit void when adapting a ULong parameter', () => {
    const type = new VoidType()
    const parameter = new UInt64Type()
    const code = generateCallback(type, parameter)
    assert.ok(code.includes('getMethod<void(jlong /* value */)>("invoke_jni")'))
    const kotlin = generateCallback(type, parameter, 'kotlin')
    assert.ok(kotlin.includes('fun invoke_jni(value: Long): Unit'))
    assert.ok(kotlin.includes('return invoke(value.toULong())'))
  })

  it('preserves nullable Long callbacks without wrapping their references twice', () => {
    const code = generateCallback(new OptionalType(new Int64Type()))
    assert.ok(
      code.includes('getMethod<jni::local_ref<jni::JLong>()>("invoke")')
    )
    assert.ok(
      code.includes(
        '__result != nullptr ? std::make_optional(__result->value()) : std::nullopt'
      )
    )
  })

  it('preserves String callback references and conversion', () => {
    const code = generateCallback(new StringType())
    assert.ok(
      code.includes('getMethod<jni::local_ref<jni::JString>()>("invoke")')
    )
    assert.ok(code.includes('return __result->toStdString();'))
  })
})
